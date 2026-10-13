#!/usr/bin/env bash
#
# Build the AgentWeave runtime environment on a CUHK CSE GPU node.
#
#   bash setup/bootstrap.sh              # build on the current node
#   bash setup/bootstrap.sh --check      # report status, change nothing
#   bash setup/bootstrap.sh --force      # rebuild the virtualenv
#
# Run it ONCE PER NODE, inside a Slurm allocation on that node, e.g.
#
#   export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
#   srun --account gpu --qos csci5540 -p csci5540 --nodelist=gpu33 \
#        --gres=gpu:1 --cpus-per-task=8 --time=1:00:00 \
#        bash setup/bootstrap.sh
#
# Idempotent: re-running only does the steps that are still missing. The
# virtualenv, model weights and JIT caches live in node-local /tmp because the
# home quota is 2.4 GB; the repository and its mini-sglang checkout live in
# $HOME, which every node shares.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SETUP_DIR}/.." && pwd)"

MODEL="${AW_MODEL:-Qwen/Qwen3-0.6B}"
SKIP_MODEL=0
FORCE=0
CHECK_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --skip-model) SKIP_MODEL=1; shift ;;
        --force) FORCE=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# shellcheck source=setup/env.sh
source "${SETUP_DIR}/env.sh"

step()  { printf '\n===== %s =====\n' "$*"; }
ok()    { printf '  [ok]   %s\n' "$*"; }
info()  { printf '  ...    %s\n' "$*"; }
fail()  { printf '  [FAIL] %s\n' "$*" >&2; exit 1; }

if [ "${CHECK_ONLY}" -eq 1 ]; then
    step "AgentWeave environment status"
    info "repo         ${AW_REPO}"
    info "node         ${AW_NODE}"
    info "node root    ${AW_NODE_ROOT}"
    aw_status || exit 1
    exit 0
fi

step "0/6  preflight"
info "repo ${AW_REPO}"
info "node ${AW_NODE}   user ${AW_USER}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "no nvidia-smi: run this inside a Slurm allocation on a GPU node, not on a login node"
fi
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 | sed 's/^/  ...    /'

# ---------------------------------------------------------------------------
# Toolchain detection.
#
# Two properties of these nodes are the reason this is detected rather than
# hardcoded, and both produce confusing errors if you get them wrong:
#
#   CUDA: flashinfer passes `--compress-mode=size` to nvcc, which only exists
#         in CUDA >= 12.9. The nodes carry 12.1 and 13.3, so 12.1 fails with
#         "nvcc fatal: Unknown option '--compress-mode=size'".
#   g++:  the default gcc/g++ 9.5 is broken - /usr/lib/gcc/x86_64-linux-gnu/9/
#         has no cc1plus - so nvcc dies with "cannot execute 'cc1plus'".
#         g++-13 is intact.
#
# NVCC_PREPEND_FLAGS is used rather than CUDAHOSTCXX alone because the JIT paths
# differ: flashinfer honours CUDAHOSTCXX, but mini-sglang's own tvm_ffi kernels
# invoke nvcc directly without -ccbin and would fall back to the broken default.
# ---------------------------------------------------------------------------

step "1/6  toolchain"

# Print the submodule commit the parent repository pins, or nothing when the
# repository is not a git checkout (e.g. unpacked from a tarball). Never fails,
# so it is safe to call from an assignment under `set -e`.
resolve_pinned_commit() {
    git -C "$1" rev-parse --git-dir >/dev/null 2>&1 || return 0
    git -C "$1" ls-tree HEAD mini-sglang 2>/dev/null | awk '{print $3}'
}

# Version comparison for dotted CUDA release numbers, e.g. version_ge 13.3 12.9.
version_ge() {
    local a_major="${1%%.*}" a_minor="${1#*.}"
    local b_major="${2%%.*}" b_minor="${2#*.}"
    a_minor="${a_minor%%.*}"
    b_minor="${b_minor%%.*}"
    if [ "${a_major}" -ne "${b_major}" ]; then
        [ "${a_major}" -gt "${b_major}" ]
        return
    fi
    [ "${a_minor}" -ge "${b_minor}" ]
}

cuda_release() {
    "$1/bin/nvcc" --version 2>/dev/null |
        sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}

# `--compress-mode` is passed straight through to the fatbin/ptxas stage, so it
# does not appear in `nvcc --help` and cannot be probed for that way. It was
# introduced in CUDA 12.9, so the release number is the reliable test.
detect_cuda() {
    local candidate release
    # Newest first.
    for candidate in $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V -r); do
        [ -x "${candidate}/bin/nvcc" ] || continue
        release="$(cuda_release "${candidate}")"
        [ -n "${release}" ] || continue
        if version_ge "${release}" "12.9"; then
            echo "${candidate} ${release}"
            return 0
        fi
    done
    return 1
}

detect_gxx() {
    local name path version
    for name in g++-15 g++-14 g++-13 g++-12 g++-11 g++; do
        path="$(command -v "${name}" 2>/dev/null)" || continue
        version="$("${path}" -dumpversion 2>/dev/null | cut -d. -f1)"
        [ -n "${version}" ] || continue
        # A compiler is only usable if nvcc can actually preprocess with it.
        if [ -e "/usr/libexec/gcc/x86_64-linux-gnu/${version}/cc1plus" ] ||
           [ -e "/usr/lib/gcc/x86_64-linux-gnu/${version}/cc1plus" ]; then
            echo "${path}"
            return 0
        fi
    done
    return 1
}

detect_cuda_out="$(detect_cuda)" \
    || fail "no CUDA toolkit >= 12.9 under /usr/local (flashinfer needs --compress-mode)"
DETECTED_CUDA="${detect_cuda_out%% *}"
DETECTED_CUDA_RELEASE="${detect_cuda_out##* }"

DETECTED_GXX="$(detect_gxx)" || fail "no working g++ (one with cc1plus) found"
DETECTED_GCC="$(dirname "${DETECTED_GXX}")/$(basename "${DETECTED_GXX}" | sed 's/^g++/gcc/')"
[ -x "${DETECTED_GCC}" ] || DETECTED_GCC="${DETECTED_GXX}"

ok "CUDA  ${DETECTED_CUDA} (release ${DETECTED_CUDA_RELEASE})"
ok "g++   ${DETECTED_GXX}"

export CUDA_HOME="${DETECTED_CUDA}"
export CUDAHOSTCXX="${DETECTED_GXX}"
export CC="${DETECTED_GCC}"
export CXX="${DETECTED_GXX}"
export NVCC_PREPEND_FLAGS="-ccbin ${DETECTED_GXX}"

GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
export TORCH_CUDA_ARCH_LIST="${GPU_CAP:-8.6}"
ok "TORCH_CUDA_ARCH_LIST ${TORCH_CUDA_ARCH_LIST}"

# ---------------------------------------------------------------------------

step "2/6  directories"
mkdir -p "${AW_NODE_ROOT}" "${AW_CACHE}" "${AW_HF}"
ok "${AW_NODE_ROOT}"

step "3/6  uv"
UV="${HOME}/.local/bin/uv"
export UV_CACHE_DIR="${AW_CACHE}/uv"
export UV_LINK_MODE=copy
if [ ! -x "${UV}" ]; then
    # Python's own venv module is unusable here: Ubuntu ships without
    # ensurepip and there is no sudo to add python3.12-venv.
    info "installing uv into ~/.local/bin"
    python3 -m pip install --user --quiet uv || fail "could not install uv"
fi
ok "$("${UV}" --version)"

step "4/6  virtualenv"
if [ "${FORCE}" -eq 1 ]; then
    info "--force: removing ${AW_VENV}"
    rm -rf "${AW_VENV}"
fi
if [ -x "${AW_PY}" ]; then
    ok "already present: ${AW_VENV}"
else
    info "creating ${AW_VENV}"
    "${UV}" venv --python /usr/bin/python3 "${AW_VENV}"
fi

step "5/6  mini-sglang and dependencies"

# Check out the exact submodule commit the parent repository pins. Cloning a
# branch would silently drift; the gitlink is the version every experiment
# result is attributed to.
PINNED="$(resolve_pinned_commit "${REPO_ROOT}")" || PINNED=""
if [ -n "${PINNED}" ]; then
    info "pinned submodule commit ${PINNED}"
else
    info "no gitlink available (not a git checkout?); tracking branch 'agentweave'"
fi

if [ -d "${AW_MINI_SGLANG}/.git" ]; then
    ok "checkout present"
else
    info "cloning into ${AW_MINI_SGLANG}"
    # HTTPS, not the SSH URL in .gitmodules: team accounts have no SSH key on
    # this cluster and the fork is public. --branch matters: the fork's default
    # branch is the upstream mirror and does not contain the AgentWeave code.
    git clone --quiet --branch agentweave \
        https://github.com/abnerhexu/mini-sglang.git "${AW_MINI_SGLANG}" \
        || fail "git clone failed"
fi
git -C "${AW_MINI_SGLANG}" fetch --all --quiet || info "fetch failed (offline?) - continuing with local refs"
if [ -n "${PINNED}" ]; then
    git -C "${AW_MINI_SGLANG}" checkout --quiet "${PINNED}" \
        || fail "cannot check out pinned commit ${PINNED}"
fi
ok "mini-sglang at $(git -C "${AW_MINI_SGLANG}" rev-parse --short HEAD)"

info "installing package and dependencies (several GB; first run takes minutes)"
"${UV}" pip install --python "${AW_PY}" -e "${AW_MINI_SGLANG}" >/dev/null || fail "dependency install failed"
"${UV}" pip install --python "${AW_PY}" --quiet ninja pytest matplotlib huggingface_hub \
    || fail "tooling install failed"
ok "dependencies installed"

step "6/6  model weights"
if [ "${SKIP_MODEL}" -eq 1 ]; then
    info "--skip-model: not downloading ${MODEL}"
elif [ -d "${AW_HF}/hub" ] && [ -n "$(ls -A "${AW_HF}/hub" 2>/dev/null)" ]; then
    ok "already cached in ${AW_HF}"
else
    info "downloading ${MODEL} into ${AW_HF}"
    "${AW_PY}" - "${MODEL}" <<'PYCODE' || fail "model download failed"
import sys
from huggingface_hub import snapshot_download
print("  ...    ", snapshot_download(sys.argv[1]))
PYCODE
fi

# ---------------------------------------------------------------------------
# Record the detected toolchain so later shells get exactly what this build
# used, instead of env.sh's best-guess fallbacks.
# ---------------------------------------------------------------------------
cat > "${AW_NODE_ROOT}/toolchain.sh" <<EOF
# Written by setup/bootstrap.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) for node ${AW_NODE}.
export CUDA_HOME=${DETECTED_CUDA}
export CUDAHOSTCXX=${DETECTED_GXX}
export CC=${DETECTED_GCC}
export CXX=${DETECTED_GXX}
export NVCC_PREPEND_FLAGS="-ccbin ${DETECTED_GXX}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
EOF

# ---------------------------------------------------------------------------
# Smoke-check the CUDA stack. This catches the two failure modes above without
# needing a GPU allocation, and it is cheap relative to everything else.
# ---------------------------------------------------------------------------
step "verify"
VERIFY_LOG="${AW_NODE_ROOT}/verify.log"
if "${AW_PY}" - >"${VERIFY_LOG}" 2>&1 <<PYCODE
import torch
for name in ("torch", "flashinfer", "sgl_kernel", "transformers", "minisgl"):
    module = __import__(name)
    print(f"  [ok]   {name:14s} {getattr(module, '__version__', '')}")
assert torch.cuda.is_available(), "CUDA is not available"
print(f"  [ok]   device {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PYCODE
then
    sed 's/^/ /' "${VERIFY_LOG}"
    ok "CUDA stack imports and sees the GPU"
else
    sed 's/^/ /' "${VERIFY_LOG}"
    fail "verification failed; full log: ${VERIFY_LOG}"
fi

step "done"
cat <<EOF
  Environment built for node ${AW_NODE}.

  Use it:
      source ${REPO_ROOT}/setup/env.sh
      python scripts/run_matrix.py --config configs/smoke.json --dry-run

  Notes:
    * /tmp is node-local. This venv exists only on ${AW_NODE}; run this script
      again (or point --nodelist=${AW_NODE}) for any other node.
    * The mini-sglang JIT compiles kernels on first use, so the first server
      start on a fresh node is slow. Later runs reuse ${AW_CACHE}.
EOF
