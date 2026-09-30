# AgentWeave 代码编写计划

## 1. 项目目标与实现边界

AgentWeave 在 Mini-SGLang 上实现一个可复现实验原型，联合观察和协调：

1. **LLM 流式解码与 sandbox 预热**：从尚未完成的 tool call 中预测工具及所需环境，提前创建 sandbox；
2. **tool 执行窗口与 KV cache 生命周期**：在相同 runtime、模型和 workload 下比较 `no-control`、`tokencake` 和 `continuum` 三种策略；
3. **端到端测量**：统一记录生成、预热、工具执行、KV 迁移和继续生成的时间线。

首版聚焦单机、多并发、TP 可用的研究原型。真实外部工具和不可信代码的生产级隔离、跨节点 KV 迁移、完整 OpenAI tool-calling 兼容性不作为首个里程碑的阻塞项。

## 2. 仓库组织与版本管理

```text
agentweave/
├── mini-sglang/                  # git submodule；核心 serving 修改
├── configs/                      # 实验和工具配置
├── benchmark/agentweave/         # workload、回放器、结果分析
├── scripts/                      # 启动、校准、实验矩阵脚本
├── tests/                        # 跨组件和端到端测试
├── docs/                         # 架构、协议、实验方法
└── CODING_PLAN.md
```

实施约定：

- 在 `mini-sglang` 子模块中创建 `agentweave` 开发分支；每个阶段先提交子模块，再在父仓库更新 gitlink。
- 发布或多人协作前，必须把子模块分支推到可访问的 Mini-SGLang fork；仅存在本地 commit 的 gitlink 无法被其他人拉取。
- 参考实现固定到调研时的版本：
  - Mini-SGLang：`9a91cfafe754aa85daee49998176275667eb58f2`
  - TokenCake：`1901ebf`
  - vLLM-Continuum：`316a587`
- TokenCake 和 Continuum 仅作为实现参考，不直接作为子模块或运行依赖；移植代码时保留原许可证和来源注释。
- 父仓库按里程碑提交，建议分支：`main`、`feature/prewarm`、`feature/kv-tiering`、`feature/policies`、`feature/benchmarks`。

## 3. 总体架构

```text
Agent API / Workload Driver
          │
          ▼
Cross-stage Coordinator (agent_id, turn_id, lifecycle)
   ├── streaming tool-call parser ──► confidence policy
   │                                      │
   │                                      ▼
   │                              Sandbox Manager / Warm Pool
   │                                      │
   │                               Python / Wasm backend
   │
   └── tool lifecycle events ─────► Mini-SGLang Scheduler
                                          │
                                  KV Policy Controller
                            ┌─────────────┼─────────────┐
                         No-Control    TokenCake     Continuum
                                          │
                              GPU KV Pool ⇄ Pinned CPU KV Pool
```

关键原则：

- coordinator 只通过明确的 lifecycle/event 协议控制 serving 和 sandbox，不把 sandbox 阻塞操作放入 GPU scheduler 主循环。
- 三种 KV 策略共享同一套 KV tiering、事件、计量和 sandbox 实现，策略只决定何时 retain/offload/prefetch/release。
- 默认 Mini-SGLang 行为保持不变；未携带 AgentWeave metadata 的普通推理请求不进入实验逻辑。

## 4. 分阶段实施

### 阶段 0：基线、可运行性与性能基准

**任务**

- 固定子模块版本，建立开发分支并记录 CUDA、PyTorch、模型和硬件信息。
- 安装 Mini-SGLang，运行现有 unit tests、离线 benchmark 和在线 streaming smoke test。
- 选择首个模型（建议 `Qwen/Qwen3-0.6B` 做功能验证，较大 Qwen 模型做性能实验）。
- 保存原始 Mini-SGLang 的吞吐、TTFT、TPOT、显存和 KV 使用率。

**验收**

- 普通 `/generate` 和 `/v1/chat/completions` 在改动前后结果一致；
- 有一份机器可读的 baseline JSON 和环境清单。

### 阶段 1：统一协议、请求身份与可观测性

**Mini-SGLang 修改点**

- 扩展 `TokenizeMsg`、`UserMsg`、`PendingReq`、`Req`，传递可选的：
  - `agent_id`、`turn_id`、`lifecycle_id`；
  - tool/sandbox 类型；
  - KV policy metadata 和预计 tool duration。
- 新增 tool lifecycle control message：
  - `tool_predicted`、`generation_finished`、`tool_started`、`tool_finished`、`continuation_submitted`、`abort`。
- lifecycle registry 必须处理乱序和重复事件，因为 `tool_started` 可能早于 scheduler 完成当前 generation 的清理。
- 实现 JSONL event logger 和 metrics collector，所有时间使用 monotonic clock，跨进程消息同时携带 wall-clock 仅用于对齐日志。

**首批指标**

- 每个 agent/turn 的 LLM 开始、首 token、末 token；
- 预测、sandbox 预热开始/完成、tool 开始/结束；
- tool 返回后的 continuation 提交和首 token；
- KV GPU/CPU bytes、D2H/H2D bytes 与耗时、命中/重算 token 数；
- policy decision、原因、预测时长、实际时长和显存压力。

**验收**

- 单 agent 多轮和并发 agent 的 event 能按 lifecycle 重建完整时间线；
- 普通请求没有 AgentWeave metadata 时零行为变化，且仅有可忽略的埋点开销。

### 阶段 2：最小端到端 Agent Runtime

**任务**

- 为 chat request 增加实验所需的 `tools`、`tool_choice` 和 agent metadata；先支持统一 JSON function-call 格式，再增加模型专用 parser。
- 实现 coordinator 驱动的循环：
  1. 提交 LLM generation；
  2. 消费流式输出并预测工具；
  3. 完整 tool call 到达后执行工具；
  4. 将 tool result 加入上下文并提交 continuation；
  5. 直到没有 tool call 或达到最大步数。
- 新增实验端点（建议 `/v1/agent/runs`）和 Python client；不要改变标准 chat endpoint 的语义。
- 首先提供 deterministic fake LLM/tool trace 和 simulated sandbox，便于无 GPU 测试跨阶段状态机。

**验收**

- ReAct/CodeAct 风格的两轮以上流程可完成；
- abort、timeout、模型输出无效 JSON、tool 失败均能清理 lifecycle 和资源。

### 阶段 3：Sandbox backend、预热与 warm pool

**模块**

- `SandboxBackend`：`prepare()`、`execute()`、`reset()`、`close()`；
- `ProcessSandboxBackend`：Python 子进程、namespace、工作目录、依赖/文件准备；
- `WasmSandboxBackend`：作为可选 backend，首版仅支持预定义 WASI 工具；
- `WarmPool`：按 `(backend, image/runtime, dependency_set, preparation_level)` 分桶；
- `PrewarmController`：流式 parser、置信度计算、压力感知阈值和状态机。

**预热状态机**

```text
DETECTING -> PREWARMING -> READY -> CLAIMED -> EXECUTING -> RECYCLED
                         └-> POOLED -> EXPIRED
任意状态 -------------------------------> FAILED/CANCELLED
```

**置信度特征**

- tool name 的完整/前缀匹配和 schema 唯一性；
- 已出现的参数 key 和参数前缀；
- 同一 agent 的历史 tool transition；
- 可用时加入 token log-probability；首版不能依赖该特征，因为当前 Mini-SGLang 不返回 logprob；
- 当前 warm-pool 容量、CPU 内存、进程数和 sandbox 创建成本。

建议决策形式为预期收益而非固定阈值：

```text
P(correct) * hidden_cold_start
  - P(wrong) * incremental_resource_cost
  - pressure_penalty > 0
```

**错误预测处理**

- 错误预热的 sandbox 完成最小清理后进入 warm pool，不立即销毁；
- 配置 per-key/global capacity、TTL、LRU eviction 和资源水位；
- sandbox 必须在复用前清理进程状态、临时文件、环境变量和未完成子进程；清理失败直接销毁。

**验收**

- 分别测出冷启动、正确预热、错误后复用、pool miss/hit 的延迟；
- 预热任务不阻塞 token streaming；
- 高资源压力下能提高阈值或拒绝预热。

### 阶段 4：Mini-SGLang KV tiering 基础设施

当前 Mini-SGLang 的 `MHAKVCache` 是单个 GPU tensor，Radix cache 只记录 GPU token indices，没有 host tier。因此不能只移植策略判断，必须先实现可迁移的数据面。

**4.1 KV pool 接口**

- 在 `BaseKVCachePool`/`MHAKVCache` 增加按 page 批量复制接口和每 page byte size；
- 建立 pinned CPU page pool，容量由 `--agent-kv-host-gb` 或 page 数配置；
- 建立独立 CUDA transfer stream、event/fence 和有界 transfer queue；
- TP 下每个 rank 保存和迁移自己的 KV shard，控制决策由 rank 0 广播，所有 rank 使用相同 page/lifecycle 操作顺序。

**4.2 Tier-aware prefix metadata**

- 为 radix prefix 增加稳定 block key（token block/hash）和 residency：`GPU`、`D2H`、`CPU`、`H2D`；
- 为 policy 增加 cache lease：同一 prefix 可同时持有 request lock 和 policy pin，二者独立释放；
- 增加定向回收 API，只回收 refcount 为 0 的叶子/后缀节点；共享祖先或正在被 request 使用的 page 不得迁移或释放；
- CPU copy 完成前保持 GPU pin；H2D 完成并写回 page table/radix metadata 前不得调度使用该 prefix；
- host pool 满时采用明确的 host eviction policy，并记录被拒绝和被驱逐原因。

**4.3 恢复路径**

- prefill admission 同时查询 GPU radix 和 host prefix index；
- host 命中时分配 GPU pages 并异步 H2D；ready 后再进入 prefill，避免读取未完成数据；
- continuation 到达前的 proactive prefetch 可先恢复已知的历史 prefix；新增 tool result 部分仍正常 prefill；
- transfer 失败、abort 或 host entry 过期时回退到正常 recomputation。

**验收**

- D2H 后释放 GPU pages，H2D 后 logits 与未迁移路径一致；
- page refcount、free list、radix metadata 在成功、取消和失败注入下保持一致；
- 覆盖 page size > 1、共享 prefix、多并发和 TP；
- 单独测量不同 KV 大小的 D2H/H2D bandwidth 和固定开销，为 TokenCake 决策提供校准参数。

### 阶段 5：三种 KV policy

定义统一接口：

```python
class ToolKVPolicy:
    def on_generation_finished(...): ...
    def on_tool_started(...): ...
    def on_tick(...): ...
    def on_tool_finished(...): ...
    def on_continuation(...): ...
    def on_abort(...): ...
```

所有策略只能调用相同的 `retain/offload/prefetch/release` 原语。

#### 5.1 No-Control

- tool lifecycle 不触发 retain、offload、prefetch 或 eviction；
- 完全使用 Mini-SGLang 默认 radix LRU 行为；
- 仍记录指标，作为严格 baseline。

#### 5.2 TokenCake 移植

- 维护按 tool type/agent type 的历史时长，可融合 client 提供的 duration estimate；
- 在线校准每个 KV page 的 D2H/H2D 时间；
- 仅当预计空闲窗口能够覆盖 `D2H + H2D + safety margin`，且 GPU 压力/等待需求足够时 offload；
- 根据 `predicted_finish - estimated_H2D - lead_time` 启动 proactive prefetch；预测提前结束时 continuation 等待剩余 H2D，预测过晚则记录过早占用 GPU 的时间；
- 加入 transfer queue、host capacity、连续迁移抖动和高价值 prefix 的保护条件。

参考 `pku-lemonade/TokenCake@1901ebf` 的 lifecycle、benefit evaluation、CPU capacity 和 transfer accounting，但根据本项目计划补上 proactive prefetch。参考仓库当前 README 明确说明其恢复仍是 demand-triggered，因此实验报告需标明差异。

#### 5.3 Continuum 移植

- 按 tool type 维护有界延迟样本、EWMA 以及可配置 quantile；
- TTL 由历史分布和 GPU opportunity cost 决定，而非硬编码常数；
- tool 开始后在 TTL 内通过 policy lease 将 KV pin 在 GPU；TTL 到期后解除 pin，使其进入默认 eviction；
- continuation 在 TTL 内返回时优先调度并释放旧 lease；
- GPU 高压时缩短 TTL，低压时允许扩大 TTL，并设置 min/max cap；
- 首版 Continuum 不主动 D2H；如需 CPU fallback，作为单独 ablation，避免混淆论文策略。

参考 `Hanchenli/vllm-continuum@316a587` 的 pin/unpin 和 continuation 优先级。该仓库当前实现使用固定 2 秒阈值且 README 标明不含 paper estimator；本项目应实现研究计划所述的历史分布 TTL，并在报告中说明这不是逐行复现。

**验收**

- 同一 trace 可仅通过 `--agent-kv-policy no-control|tokencake|continuum` 切换；
- 策略关闭时不创建 host pool 或 transfer stream；
- deterministic clock 测试覆盖短 tool、长 tool、误预测、host 满、GPU 高压和 continuation race。

### 阶段 6：Benchmark 与实验矩阵

**工作负载**

1. **可控 synthetic trace**：固定 prompt/output 长度、tool duration、sandbox cold-start、agent steps；
2. **结构化 function calling**：不同 tool name/argument 长度，测量可隐藏窗口；
3. **CodeAct/coding agent trace replay**：shell、Python、文件操作、测试命令；
4. 可选真实 coding-agent/SWE-bench 子集，用于外部有效性，不替代可重复的 trace replay。

**实验分组**

- 预热实验：`disabled`、固定阈值、confidence-aware、confidence-aware + warm pool；
- KV 主实验：固定预热机制，仅切换 `no-control`、`tokencake`、`continuum`；
- 交叉实验仅作为补充：prewarm on/off × 三种 KV policy。

**自变量**

- tool duration 分布和可预测性；
- sandbox cold-start 与 preparation level；
- prompt/history 长度和 KV cache 大小；
- 并发数/到达率；
- GPU KV 容量/显存压力；
- CPU KV 容量和 PCIe/NVLink 传输带宽；
- 预测错误率、置信度阈值、Continuum quantile/TTL。

**核心指标**

- agent end-to-end completion time（mean/P50/P95/P99）；
- sandbox cold-start hidden ratio、正确/错误预热率和 warm-pool reuse；
- tool return 到 continuation 首 token 的延迟；
- request/agent throughput、GPU utilization、GPU KV occupancy；
- D2H/H2D bytes、传输时间、重算 token、prefix hit rate；
- policy overhead、资源浪费和公平性。

**方法要求**

- 固定随机种子、模型、采样参数和 trace；
- 每个点 warm-up 后至少重复多次并输出置信区间；
- correctness 与性能实验分开；
- 每次结果记录 git commit、submodule commit、配置、硬件和环境。

### 阶段 7：测试、文档与收尾

**测试层次**

- unit：stream parser、confidence、duration estimator、TTL、warm-pool eviction、policy decisions；
- property/state-machine：lifecycle 乱序、重复、timeout、abort；
- KV correctness：page copy、hash/residency、共享 prefix、refcount/free-list invariant；
- integration：fake model + sandbox；单 GPU小模型；TP smoke test；
- regression：原 Mini-SGLang API 和 benchmark 不退化；
- fault injection：sandbox crash、D2H/H2D failure、host OOM、client disconnect。

**文档**

- `docs/architecture.md`：进程、消息和状态机；
- `docs/protocol.md`：API 和 lifecycle event schema；
- `docs/kv-tiering.md`：page ownership、同步和失败恢复；
- `docs/reproduction.md`：环境、命令和实验矩阵；
- `docs/reference-differences.md`：与 TokenCake/Continuum 参考仓库的差异。

## 5. 建议代码落点

在 Mini-SGLang 子模块中新增：

```text
python/minisgl/agentweave/
├── protocol.py
├── lifecycle.py
├── metrics.py
├── coordinator.py
├── prewarm/
│   ├── parser.py
│   ├── confidence.py
│   └── controller.py
├── sandbox/
│   ├── base.py
│   ├── process.py
│   ├── wasm.py
│   └── pool.py
└── kv/
    ├── host_pool.py
    ├── transfer.py
    ├── registry.py
    └── policy/
        ├── base.py
        ├── no_control.py
        ├── tokencake.py
        └── continuum.py
```

主要现有修改点：

- `python/minisgl/server/api_server.py`：agent endpoint、stream hook、coordinator；
- `python/minisgl/message/*.py`：metadata 和 lifecycle control messages；
- `python/minisgl/tokenizer/{server,detokenize}.py`：增量文本事件转发；
- `python/minisgl/core.py`：request metadata；
- `python/minisgl/scheduler/{scheduler,prefill,cache,config}.py`：policy hooks、host restore admission；
- `python/minisgl/kvcache/{base,mha_pool,radix_cache}.py`：page transfer、residency、lease 和定向回收；
- `python/minisgl/server/args.py`：AgentWeave、host KV 和 policy 参数；
- `tests/agentweave/`：子模块内白盒测试。

父仓库放置跨组件 workload、启动脚本、实验配置和分析代码，避免把研究数据及大规模实验产物放入 Mini-SGLang 子模块。

## 6. 里程碑与完成标准

| 里程碑 | 交付物 | 完成标准 |
|---|---|---|
| M0 | baseline + event schema | 原功能通过，时间线可重建 |
| M1 | agent loop + simulated sandbox | 无 GPU 可测完整多轮流程 |
| M2 | confidence prewarming + warm pool | 冷启动可被部分隐藏，错误预测可复用 |
| M3 | GPU/CPU KV tiering | D2H/H2D 正确、可回收、可恢复 |
| M4 | 三策略实现 | 单参数切换，策略测试完备 |
| M5 | benchmark harness | 可自动扫描四类关键自变量 |
| M6 | 完整实验与文档 | 可复现图表、差异和局限清楚 |

## 7. 优先级与风险控制

1. **先做 lifecycle 和 instrumentation，再做策略**：否则无法确认收益来自预热、KV 还是排队变化。
2. **先 simulated sandbox，再真实 sandbox**：把状态机问题与系统依赖分开。
3. **先 KV correctness，再异步优化**：第一版可同步复制验证数据布局，随后引入 transfer stream。
4. **先单 GPU，再 TP**：但接口从一开始携带 rank 一致的 page 操作，避免后期重构。
5. **严格区分 retain、evict、offload**：仅 pin/unpin 不是 CPU offload，仅复制但不释放 GPU 也不会缓解显存压力。
6. **避免比较偏差**：KV 三策略使用相同预热、workload、host pool 和 transfer primitive；No-Control 不允许隐式使用 tool-aware hint。
7. **关注参考实现差异**：当前两个参考仓库都不完全等同于研究计划中的算法描述，最终论文必须报告移植和扩展内容。
