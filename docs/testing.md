# Testing AgentWeave

## 1. CPU development checks

Use the configured `mlsys` interpreter explicitly:

```bash
cd mini-sglang
PYTHONPATH=python /home/hexu/miniconda3/envs/mlsys/bin/python \
  -m pytest -q tests/agentweave
```

The suite covers lifecycle ordering, streaming tool parsing, prewarming, warm-pool reuse, process sandbox reset, the multi-turn runtime, KV policies, host-page capacity and eviction, CPU page transfers, targeted radix eviction, continuation restore deferral/fallback, event analysis, and the benchmark client.

Run formatting and lint checks:

```bash
cd mini-sglang
/home/hexu/miniconda3/envs/mlsys/bin/python -m black --check \
  python/minisgl/agentweave python/minisgl/benchmark/agent.py \
  python/minisgl/kvcache python/minisgl/scheduler python/minisgl/server \
  tests/agentweave
/home/hexu/miniconda3/envs/mlsys/bin/python -m ruff check \
  python/minisgl/agentweave python/minisgl/benchmark/agent.py \
  python/minisgl/kvcache python/minisgl/scheduler python/minisgl/server \
  tests/agentweave
git diff --check
```

The local machine has no CUDA, so skipped CUDA tests and performance results must be run elsewhere.

## 2. CUDA correctness checks

On a CUDA machine, install Mini-SGLang in the same environment and run:

```bash
cd mini-sglang
PYTHONPATH=python python -m pytest -q tests/agentweave
```

Confirm that `test_mha_cache_copies_pages_between_cuda_and_pinned_host` runs rather than skips. Then run the upstream Mini-SGLang tests supported by that machine:

```bash
PYTHONPATH=python python -m pytest -q tests/core
```

Start with a single GPU and radix cache before testing tensor parallelism.

## 3. Launch an AgentWeave server

The following TokenCake configuration enables the Agent API, host KV pages, event logging, and demand-restore timeout:

```bash
cd mini-sglang
PYTHONPATH=python python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --cache-type radix \
  --enable-agentweave-runtime \
  --agentweave-kv-policy tokencake \
  --agentweave-host-kv-pages 4096 \
  --agentweave-continuation-restore-timeout-s 5 \
  --agentweave-event-log '../results/tokencake-{component}-{rank}.jsonl'
```

For the baselines, restart the server with exactly the same model and serving parameters, changing only the policy:

```text
--agentweave-kv-policy no-control --agentweave-host-kv-pages 0
--agentweave-kv-policy continuum  --agentweave-host-kv-pages 0
--agentweave-kv-policy tokencake  --agentweave-host-kv-pages 4096
```

Use `--agentweave-transfer-calibration-alpha 0` for a fixed transfer model, or keep the default `0.2` for online EWMA calibration. Calibration is automatically disabled for tensor-parallel runs to prevent rank-local timing noise from changing policy decisions.

## 4. Smoke-test the Agent endpoint

Create `workload.jsonl` with one JSON object per line:

```json
{"model":"Qwen/Qwen3-0.6B","agent_id":"smoke-agent","run_id":"smoke-1","messages":[{"role":"system","content":"Use the lookup tool. Emit a tool call as JSON: {\"name\":\"lookup\",\"arguments\":{\"result\":4}}. After receiving the tool result, answer normally."},{"role":"user","content":"Look up the value."}],"tools":[{"name":"lookup","required_keys":["result"],"sandbox_backend":"simulated","runtime":"lookup-delay-1","environment":{"AGENTWEAVE_SIMULATED_PREPARE_DELAY_S":"0.1","AGENTWEAVE_SIMULATED_EXECUTION_DELAY_S":"1.0"}}],"max_steps":4,"max_tokens":128}
```

The simulated delay variables allow deterministic sandbox cold-start and tool windows. Environment values are included in the warm-pool key, so sandboxes with different configurations are not mixed.

Run it:

```bash
PYTHONPATH=mini-sglang/python python -m minisgl.benchmark.agent \
  workload.jsonl --endpoint http://127.0.0.1:1919/v1/agent/runs \
  --concurrency 1 --output results/client.jsonl
```

Check that the client reports zero failures and that the server log contains `tool_predicted`, `sandbox_acquired`, `tool_started`, `tool_finished`, and KV policy decisions.

## 5. Policy comparison

Use the same workload file, request order, concurrency, model, cache size, and transfer parameters for every run. Restart the server between policies so history and warm pools begin from the same state.

Example load run:

```bash
PYTHONPATH=mini-sglang/python python -m minisgl.benchmark.agent \
  workload.jsonl --concurrency 8 --repeats 10 \
  --output results/tokencake-client.jsonl
```

Summarize scheduler events:

```bash
PYTHONPATH=mini-sglang/python python -m minisgl.agentweave.analysis \
  results/tokencake-scheduler-0.jsonl \
  --include-lifecycles --output results/tokencake-summary.json
```

Compare at least:

- client end-to-end latency and failures;
- TTFT and generation latency;
- sandbox wait, tool execution, and complete tool-window latency;
- prediction accuracy and prewarm/pool/cold acquisition counts;
- KV offload/prefetch/release actions and failures;
- D2H/H2D pages, reclaimed GPU pages, measured transfer time, and calibrated bandwidth;
- continuation restore deferred/ready/fallback events.

## 6. Required GPU stress cases

Before using results in a report, run these cases on the target CUDA machine:

1. page sizes greater than one;
2. host capacity smaller than one large prefix;
3. shared radix ancestors with two concurrent continuations;
4. continuation arriving before proactive prefetch;
5. forced H2D allocation failure and timeout fallback;
6. abort while a continuation is deferred;
7. long and short tool-duration distributions;
8. tensor parallelism with calibration disabled;
9. repeated runs with `cache_manager.check_integrity()` enabled;
10. output/logit comparison against No-Control for identical prompts and seeds.

Synchronous transfer is currently the correctness path. CUDA stream/event overlap remains a GPU-only optimization and should not be claimed until these checks pass on the target hardware.
