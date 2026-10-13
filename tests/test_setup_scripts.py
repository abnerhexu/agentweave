"""Tests for the cluster setup scripts.

`setup/` is the first thing a teammate runs, so its logic is worth pinning down
even though it is shell rather than Python. These tests exercise the version
comparison and the path derivation without needing a GPU or a cluster.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_DIR = REPO_ROOT / "setup"
BOOTSTRAP = SETUP_DIR / "bootstrap.sh"
ENV_SH = SETUP_DIR / "env.sh"


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("script", [BOOTSTRAP, ENV_SH])
def test_scripts_are_syntactically_valid(script):
    result = run_bash(f"bash -n {script}")
    assert result.returncode == 0, result.stderr


def test_bootstrap_is_executable_and_has_a_shebang():
    assert BOOTSTRAP.read_text().startswith("#!/usr/bin/env bash")
    assert BOOTSTRAP.stat().st_mode & 0o111


@pytest.mark.parametrize(
    ("version", "minimum", "expected"),
    [
        ("13.3", "12.9", True),
        ("12.9", "12.9", True),
        ("12.10", "12.9", True),
        ("12.8", "12.9", False),
        ("12.1", "12.9", False),
        ("11.8", "12.9", False),
        ("10.2", "12.9", False),
    ],
)
def test_version_ge_used_to_pick_the_cuda_toolkit(version, minimum, expected):
    """flashinfer needs `--compress-mode`, which appeared in CUDA 12.9.

    The option is passed through to the fatbin stage, so it cannot be detected
    with `nvcc --help`; the release number is the reliable test.
    """

    script = f"""
    set -e
    eval "$(sed -n '/^version_ge()/,/^}}/p' {BOOTSTRAP})"
    if version_ge "{version}" "{minimum}"; then echo yes; else echo no; fi
    """
    result = run_bash(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ("yes" if expected else "no")


def eval_bootstrap_function(name: str) -> str:
    """Extract one function definition from bootstrap.sh so it can be tested."""

    return f"""
    set -e
    eval "$(sed -n '/^{name}()/,/^}}/p' {BOOTSTRAP})"
    """


def _pinned_commit_for(path: str) -> str:
    result = run_bash(
        eval_bootstrap_function("resolve_pinned_commit") + f'resolve_pinned_commit "{path}" || true'
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_resolve_pinned_commit_reads_the_gitlink():
    """A git checkout must resolve to the pinned submodule commit."""

    assert _pinned_commit_for(REPO_ROOT) == "8e91ad2c614da25035d239e4967311110c3d2404"


def test_resolve_pinned_commit_is_empty_for_a_non_git_directory(tmp_path):
    """Regression: a tarball checkout made bootstrap abort under `set -e`.

    The failing `git ls-tree` propagated through the command substitution and
    killed the script before it reached the clone step.
    """

    assert _pinned_commit_for(str(tmp_path)) == ""


def test_resolve_pinned_commit_does_not_abort_the_caller(tmp_path):
    """Calling it in an assignment must leave the script alive."""

    result = run_bash(eval_bootstrap_function("resolve_pinned_commit") + f"""
        PINNED="$(resolve_pinned_commit "{tmp_path}")" || PINNED=""
        echo "survived: '${{PINNED}}'"
        """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "survived: ''"


def test_clone_pins_the_agentweave_branch():
    """The fork's default branch is the upstream mirror with no AgentWeave code.

    Cloning without --branch would silently produce a tree with no
    agentweave package in it.
    """

    text = BOOTSTRAP.read_text()
    clone_lines = [
        line for line in text.splitlines() if "git clone" in line or "--branch agentweave" in line
    ]
    assert any("--branch agentweave" in line for line in clone_lines), clone_lines


def test_env_sh_derives_every_path_from_user_and_node():
    script = f"""
    export AW_USER=alice
    export AW_NODE=gpu33
    source {ENV_SH}
    echo "$AW_ROOT|$AW_NODE_ROOT|$AW_VENV|$AW_PY|$AW_CACHE"
    """
    result = run_bash(script)
    assert result.returncode == 0, result.stderr
    root, node_root, venv, python, cache = result.stdout.strip().split("|")
    assert root == "/tmp/aw-alice"
    # Node level is present because /tmp is node-local.
    assert node_root == "/tmp/aw-alice/gpu33"
    assert venv == "/tmp/aw-alice/gpu33/venv"
    assert python == "/tmp/aw-alice/gpu33/venv/bin/python"
    assert cache == "/tmp/aw-alice/gpu33/cache"


def test_env_sh_puts_the_venv_first_on_path():
    script = f"""
    export AW_USER=alice AW_NODE=gpu33
    source {ENV_SH}
    echo "$PATH" | tr ':' '\\n' | head -1
    """
    result = run_bash(script)
    assert result.stdout.strip() == "/tmp/aw-alice/gpu33/venv/bin"


def test_env_sh_keeps_jit_caches_out_of_home():
    """Home has a 2.4 GB quota; both JIT engines default to $HOME/.cache."""

    script = f"""
    export AW_USER=alice AW_NODE=gpu33 HOME=/uac/msc/alice
    source {ENV_SH}
    echo "$FLASHINFER_CACHE_DIR|$TVM_FFI_CACHE_DIR|$HF_HOME"
    """
    result = run_bash(script)
    flashinfer, tvm_ffi, hf = result.stdout.strip().split("|")
    for path in (flashinfer, tvm_ffi, hf):
        assert path.startswith("/tmp/aw-alice/"), path
        assert not path.startswith("/uac/msc/alice"), path


def test_env_sh_respects_pre_existing_overrides():
    """A caller that already set a path must win, so overrides compose."""

    script = f"""
    export AW_USER=alice AW_NODE=gpu33 AW_ROOT=/scratch/elsewhere
    source {ENV_SH}
    echo "$AW_ROOT|$AW_NODE_ROOT"
    """
    result = run_bash(script)
    assert result.stdout.strip() == "/scratch/elsewhere|/scratch/elsewhere/gpu33"


def test_aw_status_reports_a_missing_environment():
    script = f"""
    export AW_USER=nobody-$$ AW_NODE=node-that-does-not-exist
    source {ENV_SH}
    aw_status
    echo "rc=$?"
    """
    result = run_bash(script)
    assert "missing" in result.stdout
    assert "rc=1" in result.stdout


def test_env_sh_sources_a_recorded_toolchain_when_present(tmp_path):
    """bootstrap.sh records the detected toolchain; env.sh must prefer it."""

    node_root = tmp_path / "aw-test-node"
    node_root.mkdir()
    (node_root / "toolchain.sh").write_text(
        "export CUDA_HOME=/usr/local/cuda-99.9\n"
        "export NVCC_PREPEND_FLAGS='-ccbin /opt/other/g++-99'\n"
    )
    script = f"""
    export AW_USER=alice AW_NODE=testnode AW_NODE_ROOT={node_root}
    source {ENV_SH}
    echo "$CUDA_HOME|$NVCC_PREPEND_FLAGS"
    """
    result = run_bash(script)
    assert result.stdout.strip() == "/usr/local/cuda-99.9|-ccbin /opt/other/g++-99"
