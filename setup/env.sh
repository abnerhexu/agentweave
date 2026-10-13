# AgentWeave cluster environment.
#
#   source setup/env.sh
#
# This file only *computes* paths - it never creates anything, so it is always
# correct and safe to source. Run `setup/bootstrap.sh` to create the
# directories it points at.
#
# Everything is derived from $USER and the current hostname, so every team
# member gets their own tree with no editing and no configuration file:
#
#   /tmp/aw-<user>/<node>/venv   virtualenv            (node-local, ~9.5 GB)
#   /tmp/aw-<user>/<node>/hf     HuggingFace weights   (node-local, ~1.5 GB)
#   /tmp/aw-<user>/<node>/cache  flashinfer/tvm JIT    (node-local, grows)
#
# The per-node level is not decoration: /tmp is node-local, so a virtualenv
# built on gpu33 simply does not exist on gpu35. Keeping the node in the path
# makes that visible instead of silently using a half-populated directory.
# The repository itself is small and lives in $HOME, which is shared by every
# node, so it is checked out once.

# --- identity -------------------------------------------------------------
export AW_USER="${AW_USER:-$(id -un)}"
# `hostname -s` is not on every minimal image; fall back to truncating at the
# first dot.
if [ -z "${AW_NODE:-}" ]; then
    AW_NODE="$(hostname -s 2>/dev/null || hostname)"
    export AW_NODE="${AW_NODE%%.*}"
fi

# --- locations ------------------------------------------------------------
export AW_ROOT="${AW_ROOT:-/tmp/aw-${AW_USER}}"
export AW_NODE_ROOT="${AW_NODE_ROOT:-${AW_ROOT}/${AW_NODE}}"
export AW_VENV="${AW_VENV:-${AW_NODE_ROOT}/venv}"
export AW_PY="${AW_PY:-${AW_VENV}/bin/python}"
export AW_HF="${AW_HF:-${AW_NODE_ROOT}/hf}"
export AW_CACHE="${AW_CACHE:-${AW_NODE_ROOT}/cache}"

# --- repository -----------------------------------------------------------
# Resolved from this file's own location, so a checkout can live anywhere.
AW_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AW_REPO="${AW_REPO:-$(cd "${AW_SETUP_DIR}/.." && pwd)}"
export AW_MINI_SGLANG="${AW_MINI_SGLANG:-${AW_REPO}/mini-sglang}"

# --- toolchain ------------------------------------------------------------
# bootstrap.sh detects the working combination and records it here; these are
# only the fallbacks for an environment that was built before detection was
# added. See setup/bootstrap.sh for why each is needed.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-13}"
export CC="${CC:-/usr/bin/gcc-13}"
export CXX="${CXX:-/usr/bin/g++-13}"
# Covers every nvcc invocation, including libraries that do not honour
# CUDAHOSTCXX (mini-sglang's own tvm_ffi kernels shell out without -ccbin).
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin ${CUDAHOSTCXX}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# bootstrap.sh detects the toolchain that actually works on this node and writes
# it here. Prefer that over the fallbacks above.
if [ -f "${AW_NODE_ROOT}/toolchain.sh" ]; then
    # shellcheck source=/dev/null
    . "${AW_NODE_ROOT}/toolchain.sh"
fi

# --- caches ---------------------------------------------------------------
# Both JIT engines default to $HOME/.cache, and the home quota is 2.4 GB.
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${AW_CACHE}/flashinfer}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${AW_CACHE}/tvm-ffi}"
export HF_HOME="${HF_HOME:-${AW_HF}}"

# --- executables ----------------------------------------------------------
export PATH="${AW_VENV}/bin:${HOME}/.local/bin:${PATH}"

# --- status ---------------------------------------------------------------
# Prints what is missing for the current node. Called by bootstrap.sh and
# available interactively as `aw_status`.
aw_status() {
    local missing=0
    if [ ! -x "${AW_PY}" ]; then
        echo "  [missing] virtualenv for node ${AW_NODE}: ${AW_VENV}"
        echo "            -> run: bash ${AW_REPO}/setup/bootstrap.sh"
        missing=1
    fi
    if [ ! -d "${AW_MINI_SGLANG}/python/minisgl" ]; then
        echo "  [missing] mini-sglang checkout: ${AW_MINI_SGLANG}"
        echo "            -> run: bash ${AW_REPO}/setup/bootstrap.sh"
        missing=1
    fi
    if [ ! -d "${AW_HF}/hub" ]; then
        echo "  [missing] model weights in ${AW_HF} (bootstrap downloads them)"
        missing=1
    fi
    if [ "${missing}" -eq 0 ]; then
        echo "  [ok] node ${AW_NODE}: venv, mini-sglang and model cache all present"
    fi
    return "${missing}"
}

# Warn on a compute node that has no environment, rather than letting the user
# hit a confusing import error later. Silent on login nodes (which have no GPU
# and legitimately have no virtualenv) and when AW_QUIET is set, so `--dry-run`
# and workload generation from a login node stay clean.
if [ -z "${AW_QUIET:-}" ] && [ ! -x "${AW_PY}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "AgentWeave: node '${AW_NODE}' has no virtualenv yet." >&2
    echo "  Build it once on this node with:" >&2
    echo "    bash ${AW_REPO}/setup/bootstrap.sh" >&2
fi
