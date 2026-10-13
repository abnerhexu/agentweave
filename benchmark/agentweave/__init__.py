"""AgentWeave experiment harness.

The serving implementation lives in the ``mini-sglang`` submodule, which is
Linux/CUDA-only. Everything in this package is plain Python so the harness can
be developed and unit-tested on a CPU-only machine; only ``run_experiment``
needs the submodule importable because it launches the server and the workload
client.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_mini_sglang_path(mini_sglang_dir: str | Path | None = None) -> Path:
    """Put the submodule's ``python/`` directory on ``sys.path``.

    Importing ``minisgl`` only works when ``mini-sglang/python`` is importable.
    The submodule is not installed as a package (it is Linux/CUDA-only), so
    entry points call this instead of requiring the caller to export
    ``PYTHONPATH``. ``mini_sglang_dir`` may be either the submodule root or its
    ``python/`` directory.
    """

    root = (_REPO_ROOT / "mini-sglang") if mini_sglang_dir is None else Path(mini_sglang_dir)
    python_dir = (root / "python") if (root / "python").is_dir() else root
    python_dir = python_dir.resolve()
    if not (python_dir / "minisgl").is_dir():
        raise FileNotFoundError(
            f"mini-sglang source not found under {python_dir}. "
            "Initialize the submodule or pass the correct --mini-sglang-dir."
        )
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    return python_dir
