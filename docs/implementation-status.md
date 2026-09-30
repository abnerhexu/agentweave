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

## Intentionally incomplete

- TokenCake decisions are implemented, but scheduler-side D2H/H2D execution is not yet connected to the host page pool. The scheduler reports such actions with `implemented=false` rather than silently treating them as completed.
- Host-prefix matching and H2D restoration during prefill admission remain to be implemented.
- CUDA streams/events, pinned-memory bandwidth calibration, TP synchronization, and GPU correctness/performance tests require a CUDA development machine.
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

GPU results must not be inferred from these CPU tests.
