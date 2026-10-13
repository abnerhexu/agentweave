# AgentWeave experiment harness

Tools for running the experiments described in `CODING_PLAN.md` section 6 and in
the project proposal: a workload generator, a per-cell server/benchmark driver, a
matrix sweep, and a report with figures.

```
agentweave/
├── mini-sglang/                 # submodule: the serving implementation under test
├── setup/
│   ├── env.sh                   # source this; computes all paths from $USER + hostname
│   └── bootstrap.sh             # one-time per-node environment build on the cluster
├── configs/                     # experiment definitions (smoke, kv_main, prewarm)
├── benchmark/agentweave/
│   ├── workload.py              # generates the JSONL workload
│   ├── report.py                # joins results into a table + JSON
│   └── plots.py                 # PNG figures
├── scripts/
│   ├── run_experiment.py        # runs ONE cell: server + client + manifest
│   └── run_matrix.py            # sweeps a config's matrix, skipping finished cells
├── docs/gpu-cluster.md          # VPN, Slurm, environment build, troubleshooting
└── tests/                       # CPU-only tests for the harness itself
```

Everything in `benchmark/` and `scripts/` is plain Python and runs anywhere. Only
running an actual experiment needs the submodule, and the submodule needs
**Linux + CUDA** (`flashinfer`, `sgl_kernel`); it cannot start on macOS.

**New to the cluster? Start with [docs/gpu-cluster.md](docs/gpu-cluster.md).**
It covers access, the one-time build, and the failure modes that are specific to
these nodes.

## 1. Set up the submodule

The submodule is pinned by the parent repo and holds the serving implementation
under test. The upstream fork is public, so it clones over HTTPS:

```bash
git submodule update --init --recursive
```

If that fails because `.gitmodules` uses an SSH URL and your key is not on the
account:

```bash
git config submodule.mini-sglang.url https://github.com/abnerhexu/mini-sglang.git
git submodule update --init --recursive
```

On the cluster you do not need this step: `setup/bootstrap.sh` clones the
submodule itself and checks out the exact pinned commit, so results stay
attributable to a known revision.

## 2. Check the harness itself (no GPU needed)

These tests import parts of the submodule, so the submodule must be present, but
they never touch CUDA:

```bash
python -m pytest -q
```

For the serving-side tests:

```bash
cd mini-sglang && PYTHONPATH=python python -m pytest -q -o addopts="" tests/agentweave
```

## 3. Getting a GPU machine (CUHK CSE)

The CSE login and GPU nodes are only reachable over the **CSE VPN** (not the
CUHK central VPN), and the course allocation lives on a Slurm partition. The
full walkthrough — VPN setup, Slurm commands, and the one-time per-node
environment build — is in **[docs/gpu-cluster.md](docs/gpu-cluster.md)**.

Short version, once you are on the VPN and a login node:

```bash
cd ~/agentweave
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf

# one-time, per node (idempotent)
srun --account gpu --qos csci5540 -p csci5540 --nodelist=gpu33 \
     --gres=gpu:1 --cpus-per-task=8 --time=1:00:00 \
     bash setup/bootstrap.sh

source setup/env.sh
```

`setup/bootstrap.sh` auto-detects the working CUDA toolkit and host compiler,
and puts the virtualenv, model weights and JIT caches on node-local `/tmp`
(the home quota is 2.4 GB). `setup/env.sh` derives every path from `$USER` and
the hostname, so nothing needs editing and each team member gets their own
tree. Because `/tmp` is node-local, pin jobs with `--nodelist=gpu33` — or run
the bootstrap on the other node too.

`aw_status` prints what is present or missing for the current node.

## 4. Run experiments (GPU machine)

Source the environment first (`source setup/env.sh`), then pick the interpreter
that has the submodule's dependencies installed and pass it as `--python`;
`$AW_PY` already points at it.

### Plan a sweep without running it

```bash
python scripts/run_matrix.py --config configs/smoke.json --dry-run
# (workload generation and --dry-run work on any machine, including a login node)
```

This prints every cell (policy, variant, concurrency), the run directory it would
write, and the host KV pages it would configure. It creates nothing on disk.

### Run the smoke test first

```bash
python scripts/run_matrix.py --config configs/smoke.json --python "$AW_PY"
```

Four agents, one tool call each, all three KV policies. Use it to confirm the
agent endpoint works, the client reports zero failures, and the event log
contains `tool_predicted`, `sandbox_acquired` and `kv_policy_decision`. The smoke
config produces no publishable numbers: with that little load the GPU is never
under pressure, so TokenCake stays at its `low_pressure` decision.

### Then the real sweeps

```bash
python scripts/run_matrix.py --config configs/kv_main.json --python "$AW_PY"
python scripts/run_matrix.py --config configs/prewarm.json --python "$AW_PY"
```

`kv_main` is the KV-policy comparison (prewarming held fixed, sweeping
concurrency and tool duration). `prewarm` is the prewarming comparison (KV policy
held at `no-control`, sweeping the three prewarm arms and concurrency).

Each cell writes a run directory:

```
results/<timestamp>/<config>/<policy>/<variant>/c<N>/
├── manifest.json                 # cell definition, argv, git shas, GPU
├── client.jsonl                  # per-agent end-to-end results
├── server.log                    # server stdout/stderr
└── events-scheduler-0.jsonl      # the serving timeline
```

Re-running the same command resumes: cells that already have a `manifest.json`
are skipped. Use `--force` to redo them.

To run a single cell by hand:

```bash
python scripts/run_experiment.py --config configs/kv_main.json \
  --policy tokencake --variant default --concurrency 8 \
  --workload results/<timestamp>/kv-main/workload.jsonl \
  --run-dir /tmp/cell --python "$AW_PY"
```

## 5. Report

```bash
python -m benchmark.agentweave.report results/<timestamp>/kv-main \
  --output results/report.json --plot-dir results/figures
```

The table is printed to stdout and is also the non-color view of the figures
(one series color sits below 3:1 contrast on a light surface, so the numbers must
be readable as text). Figures written:

| Figure | Shows |
|---|---|
| `fig_e2e_latency.png` | agent end-to-end latency, P50/P95 |
| `fig_continuation_gap.png` | tool return to continuation first token, P50/P95 |
| `fig_prewarm.png` | sandbox acquisition mix (prewarm/pool/cold) and prediction accuracy |
| `fig_kv_transfers.png` | D2H/H2D pages and KV policy action counts |

The continuation gap is the metric that spans two lifecycles and is therefore not
in `minisgl.agentweave.analysis`; see the `continuation_gaps_ms` docstring.

## 6. Two things the harness enforces for you

**TokenCake needs a host tier.** `--agentweave-host-kv-pages` must be non-zero for
`tokencake`, otherwise its offload/prefetch actions are recorded with
`implemented=False` and the run looks like "the policy chose not to transfer"
rather than "the transfer could not run". `build_server_argv` raises instead, and
forces the value to 0 for `no-control` and `continuum` so they cannot borrow the
host tier and be compared unfairly.

**Deterministic sandbox timing needs the simulated backend.** The generator
defaults to `sandbox_backend: "simulated"` and writes the cold start and tool
duration into `AGENTWEAVE_SIMULATED_PREPARE_DELAY_S` /
`AGENTWEAVE_SIMULATED_EXECUTION_DELAY_S`. The default backend is
`python-process`, which is not deterministic and would make cell-to-cell
comparisons meaningless.

## 7. Known limitations

- **`prewarm-off` is an approximation.** There is no flag that disables
  prewarming. The config sets `prewarm_threshold: 1.0`, but `ConfidencePolicy`
  caps the threshold at `min(0.95, ...)`, and a fully-resolved tool call with
  learned transition history can reach 0.95. A true "disabled" arm needs a
  small submodule change, coordinated with the submodule owner.
- **No CodeAct / coding-agent trace replay yet.** `CODING_PLAN.md` section 6 lists
  four workload families; the harness implements the first two (controlled
  synthetic and structured function calling). Trace replay needs a real trace
  source and is not implemented.
- **Model behaviour is not fully controlled.** The generator shapes the prompt,
  but whether the model emits the intended tool call is up to the model. Check
  `client.jsonl` (`agent_status`) and the prediction accuracy figure before
  trusting a run.
- **TokenCake transfers are synchronous.** CUDA stream/event overlap is still an
  unimplemented optimization, so the measured transfer cost is the correctness
  path, not the final one.
