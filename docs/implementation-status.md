# AgentWeave Implementation Status

## Implemented locally

### Request and lifecycle control plane

- Agent identity (`agent_id`, `turn_id`, `lifecycle_id`) is propagated through API, tokenizer, scheduler, and request state.
- Tool lifecycle events are accepted at `/v1/agentweave/events`.
- Scheduler-side lifecycle tracking tolerates duplicate and out-of-order events.
- JSONL event logging records generation and tool/KV policy timestamps.

### Agent and sandbox runtime

- `/v1/agent/runs` drives multi-turn generation/tool/continuation loops when the server is launched with `--enable-agentweave-runtime`.
- Streaming JSON tool-call parsing and confidence-aware prewarming are implemented.
- Adaptive thresholds account for process, memory, and pool pressure.
- Wrong predictions are reset and placed in a keyed warm pool.
- Simulated and persistent Python-process backends are available.
- The Python-process backend is a research mechanism, not a security boundary.

### KV policy and storage foundations

- Common policy interface and deterministic implementations of:
  - No-Control;
  - TokenCake duration/transfer break-even decisions and prefetch timing;
  - Continuum historical-quantile TTL decisions.
- Continuum GPU prefix leases are connected to the Mini-SGLang scheduler.
- CPU-testable host KV page pool with capacity, pinning, LRU eviction, and partial restore.
- MHA KV cache page-copy interfaces.
- Targeted radix eviction of an unshared, unlocked leaf suffix.
- TokenCake scheduler execution path for synchronous D2H offload, proactive/demand H2D prefetch, prefix reinsertion before continuation admission, and lifecycle cleanup. Enable it with a non-zero `--agentweave-host-kv-pages` value.
- Configurable D2H/H2D cost parameters and EWMA calibration from observed transfer durations; transfer timing and modeled bandwidth are included in policy event logs.
- Continuations are deferred while demand restore is retried, then admitted when ready or sent through normal prefix recomputation after a configurable timeout/idle fallback.

## Intentionally incomplete

- TokenCake transfers currently use a synchronous correctness path. CUDA stream/event overlap remains to be implemented and validated on a GPU machine.
- TP synchronization and GPU correctness/performance tests require a CUDA development machine.
- Wasm sandbox support remains pending.

## Local validation

The local machine is used only for CPU development. Current validation command:

```bash
cd mini-sglang
PYTHONPATH=python /home/hexu/miniconda3/envs/mlsys/bin/python -m pytest -q tests/agentweave
```

Static checks:

```bash
/home/hexu/miniconda3/envs/mlsys/bin/python -m ruff check <changed paths>
/home/hexu/miniconda3/envs/mlsys/bin/python -m black --check <changed paths>
```

Run a reproducible Agent API workload (one `AgentRunRequest` JSON object per line, with optional `_delay_s`) with:

```bash
PYTHONPATH=mini-sglang/python /home/hexu/miniconda3/envs/mlsys/bin/python \
  -m minisgl.benchmark.agent workload.jsonl --concurrency 8 \
  --output client-results.jsonl
```

Use the same workload, concurrency, and initial transfer parameters for No-Control, TokenCake, and Continuum runs.

Summarize one or more scheduler JSONL logs with:

```bash
PYTHONPATH=mini-sglang/python /home/hexu/miniconda3/envs/mlsys/bin/python \
  -m minisgl.agentweave.analysis events.jsonl --output summary.json
```

The summary reports TTFT, generation/tool/sandbox timing distributions, sandbox acquisition sources, KV actions, transferred pages, failures, and measured transfer time.

See [`testing.md`](testing.md) for CPU checks, CUDA correctness tests, server launch commands, workload examples, and the policy-comparison procedure.

GPU results must not be inferred from these CPU tests.
