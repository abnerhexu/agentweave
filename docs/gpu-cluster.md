# Running AgentWeave on the CUHK CSE GPU cluster

Everything here has been run and verified on the course allocation. Follow it in
order; the only step that takes real time is the one-time environment build.

## 1. Get on the network

The CSE server hosts are unreachable from anywhere outside the CSE network —
including from the general CUHK campus network. Measured from a campus address,
`linux10` answered nothing at all (ports 22, 80, 443, 8080, 2222 all silent,
ICMP 100% loss) while `www.cse` pinged normally. A failed `ping` therefore does
**not** mean a host is down.

Connect the **CSE VPN**, not the CUHK central VPN:

- A profile for `openvpn2.cse.cuhk.edu.hk` may already be imported in *OpenVPN
  Connect*. If not, download one from <https://openvpn.cse.cuhk.edu.hk>
  (username: CSE username, password: Linux account password). The enhanced
  variant at <https://openvpne.cse.cuhk.edu.hk> additionally requires Duo 2FA,
  where the password field is your Linux password immediately followed by the
  6-digit one-time code.
- Confirm the tunnel is up — some `utun*` interface must have an address:
  ```bash
  ifconfig | grep -A2 '^utun' | grep inet
  ```

Then SSH to any login node, `linux1` through `linux15`:

```bash
ssh <cse-username>@linux1.cse.cuhk.edu.hk
```

If your key is not on the cluster yet, install it once (prompts for your
password):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <cse-username>@linux1.cse.cuhk.edu.hk
```

## 2. The course GPU allocation

The department reserves `gpu33` and `gpu35` for CSCI5540: **8x RTX 3090 (24 GB,
compute capability 8.6)** each, managed by Slurm. Set the config path in every
session — the default Slurm configuration does not know about these nodes:

```bash
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
```

```bash
sinfo -p csci5540                  # 2 nodes, 3 h time limit, gpu:rtx3090:8
squeue -u $USER                    # your jobs

# interactive GPU shell
srun --account gpu --qos csci5540 -p csci5540 --gres=gpu:1 --cpus-per-task=8 \
     --pty /bin/bash
```

The course partition is hidden, so plain `sinfo` does not list it; use
`sinfo -a` or `sinfo -p csci5540`. **The QOS allows one job per user at a
time**, so matrix sweeps run their cells sequentially.

Older department documentation claims `gpu33`/`gpu34` carry Titan Xp (sm_61)
GPUs. That was true years ago and is now wrong — sm_61 could not run this
submodule at all, since `flashinfer` and `sgl_kernel` need sm_80+.

## 3. One-time environment build (per node)

`/tmp` is node-local on these machines, so an environment built on `gpu33` does
not exist on `gpu35`. Build it once on each node you intend to use.

```bash
cd ~/agentweave                       # this repository
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
srun --account gpu --qos csci5540 -p csci5540 --nodelist=gpu33 \
     --gres=gpu:1 --cpus-per-task=8 --time=1:00:00 \
     bash setup/bootstrap.sh
```

It is idempotent — re-running only does what is still missing. `--force`
rebuilds the virtualenv, `--check` reports status without changing anything,
`--skip-model` leaves the weights alone.

About 10 GB is downloaded and written; expect a few minutes.

### Why the build is shaped this way

These four properties of the nodes cost real debugging time, and the scripts
handle them automatically:

| Property | Consequence | Handled by |
|---|---|---|
| Home quota is **2.4 GB** | torch plus its CUDA wheels do not fit | venv, weights and caches go to node-local `/tmp` (`/tmp/aw-$USER/$NODE`) |
| `python3 -m venv` fails (no `ensurepip`, no sudo) | cannot create a virtualenv the normal way | `uv`, installed into `~/.local/bin` |
| flashinfer passes `--compress-mode=size` to nvcc, which needs **CUDA >= 12.9** | node's CUDA 12.1 dies with `nvcc fatal: Unknown option` | `bootstrap.sh` picks a toolkit that supports it (13.3 here) |
| Default `gcc`/`g++` **9.5 is broken** (no `cc1plus`) | nvcc dies with `cannot execute 'cc1plus'` | detection picks e.g. `g++-13`, exported through `NVCC_PREPEND_FLAGS` so *every* nvcc call uses it |

`NVCC_PREPEND_FLAGS` rather than `CUDAHOSTCXX` alone matters because the JIT
paths differ: flashinfer honours `CUDAHOSTCXX`, but mini-sglang's own `tvm_ffi`
kernels invoke nvcc directly without `-ccbin` and would otherwise fall back to
the broken default compiler.

## 4. Use the environment

```bash
source ~/agentweave/setup/env.sh
```

That computes every path from `$USER` and the current hostname — no config file
to edit, nothing to regenerate. If something is missing for the current node it
says so instead of failing later with an import error. `aw_status` prints a
summary.

```bash
aw_status
python scripts/run_matrix.py --config configs/smoke.json --dry-run
```

Then, inside a GPU allocation:

```bash
python scripts/run_matrix.py --config configs/smoke.json
```

See the top-level `README.md` for what the configs mean and how to read the
report.

## 5. Troubleshooting

**`nvcc fatal: Unknown option '--compress-mode=size'`**
`CUDA_HOME` points at a toolkit older than 12.9. Re-run `setup/bootstrap.sh`, or
`source setup/env.sh` to pick up the recorded toolchain.

**`gcc: fatal error: cannot execute 'cc1plus'`**
Something invoked nvcc without going through `NVCC_PREPEND_FLAGS`. Check that
`env.sh` is sourced (`echo $NVCC_PREPEND_FLAGS`).

**Import error mentioning a path under `/tmp/aw-$USER/` for a different node**
You submitted to the other node. Add `--nodelist=` to pin it, or run
`setup/bootstrap.sh` there too.

**`QOSMaxSubmitJobPerUserLimit`**
Only one job per user at a time on this QOS. Wait for the running one, or
`scancel` it. Note that after `scancel` a job can sit in `CG` (completing) for a
while if the server left child processes behind — the harness now starts the
server in its own process group precisely to avoid that.

**Model answers without calling any tool (`tool_calls: 0`)**
Qwen3 runs in thinking mode by default (the submodule does not pass
`enable_thinking=False`) and will happily answer a vague prompt without using a
tool — the original `"Run task N"` prompt produced a `<think>` block and zero
tool calls. The generator now states the exact JSON shape and names the
arguments; keep that property if you change the prompts.

## 6. What lives where

| Path | Contents | Lifetime |
|---|---|---|
| `~/agentweave` | this repository + the `mini-sglang` checkout | persistent (NFS home) |
| `/tmp/aw-$USER/$NODE/venv` | virtualenv (~9.5 GB) | per node, survives between jobs |
| `/tmp/aw-$USER/$NODE/hf` | model weights | per node |
| `/tmp/aw-$USER/$NODE/cache` | flashinfer / tvm_ffi JIT caches | per node, grows on first use |

`/tmp` survives between Slurm jobs on these machines, so the build is a one-time
cost per node, not per job.
