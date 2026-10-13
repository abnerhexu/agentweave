#!/usr/bin/env python3
"""Sweep an experiment matrix, running one server-per-cell via run_experiment.

The workload file is generated once per experiment and shared by every cell:
comparing policies on different workloads would confound the result. Cells
whose ``manifest.json`` already exists are skipped, so an interrupted sweep can
be resumed by re-running the same command.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_experiment import ExperimentError, effective_host_kv_pages, run_cell

from benchmark.agentweave import ensure_mini_sglang_path
from benchmark.agentweave.workload import (
    build_workload,
    load_config,
    workload_config_from_experiment,
    write_workload,
)


def plan_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand ``matrix`` into the list of cells to run, in execution order."""

    matrix = config.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ExperimentError("config requires a 'matrix' object")
    policies = matrix.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ExperimentError("matrix.policies must be a non-empty list")
    concurrencies = matrix.get("concurrencies", [1])
    if not isinstance(concurrencies, list) or not concurrencies:
        raise ExperimentError("matrix.concurrencies must be a non-empty list")
    variants = matrix.get("server_variants") or [{"name": "default", "overrides": {}}]
    if not isinstance(variants, list):
        raise ExperimentError("matrix.server_variants must be a list")

    cells = []
    for policy, variant, concurrency in itertools.product(policies, variants, concurrencies):
        cells.append(
            {
                "policy": policy,
                "variant": variant.get("name", "default"),
                "overrides": dict(variant.get("overrides", {})),
                "concurrency": int(concurrency),
            }
        )
    return cells


def cell_dir(root: Path, cell: Mapping[str, Any]) -> Path:
    return root / str(cell["policy"]) / str(cell["variant"]) / f"c{cell['concurrency']}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep an AgentWeave experiment matrix")
    parser.add_argument("--config", required=True, help="Experiment config JSON")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Destination root (default: results/<timestamp>/<config name>).",
    )
    parser.add_argument("--policies", nargs="*", default=None, help="Subset of policies")
    parser.add_argument("--concurrencies", nargs="*", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None, help="Subset of variant names")
    parser.add_argument("--repeats", type=int, default=None, help="Override matrix.repeats")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mini-sglang-dir", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--client-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells that already have a manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned cells and exit (no server is launched).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    matrix = dict(config.get("matrix", {}))
    if args.repeats is not None:
        matrix["repeats"] = args.repeats
    config = {**config, "matrix": matrix}
    repeats = int(matrix.get("repeats", 1))

    cells = plan_cells(config)
    if args.policies:
        cells = [cell for cell in cells if cell["policy"] in set(args.policies)]
    if args.concurrencies:
        cells = [cell for cell in cells if cell["concurrency"] in set(args.concurrencies)]
    if args.variants:
        cells = [cell for cell in cells if cell["variant"] in set(args.variants)]
    if not cells:
        parser.error("no cells selected")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = (
        Path(args.results_root)
        if args.results_root
        else _REPO_ROOT / "results" / stamp / str(config.get("name", "experiment"))
    )
    workload_path = root / "workload.jsonl"

    planned = [
        {
            "policy": cell["policy"],
            "variant": cell["variant"],
            "concurrency": cell["concurrency"],
            "run_dir": str(cell_dir(root, cell)),
            "host_kv_pages": effective_host_kv_pages(
                {**dict(config.get("server", {})), **cell["overrides"]}, cell["policy"]
            ),
        }
        for cell in cells
    ]
    if args.dry_run:
        # Dry runs must not touch the filesystem, so report the intended paths
        # without creating the directory or the workload.
        print(
            json.dumps(
                {"root": str(root), "workload": str(workload_path), "cells": planned},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    root.mkdir(parents=True, exist_ok=True)
    # One workload shared by every cell so policy comparisons are fair.
    write_workload(build_workload(workload_config_from_experiment(config)), workload_path)

    if args.mini_sglang_dir:
        mini_sglang_dir = Path(args.mini_sglang_dir)
    else:
        mini_sglang_dir = ensure_mini_sglang_path().parent
    print(json.dumps({"root": str(root), "workload": str(workload_path), "cells": len(cells)}))

    failures = 0
    for index, cell in enumerate(cells, 1):
        target = cell_dir(root, cell)
        manifest_path = target / "manifest.json"
        if manifest_path.exists() and not args.force:
            print(f"[{index}/{len(cells)}] skip {target} (manifest exists)")
            continue
        print(f"[{index}/{len(cells)}] run {target}")
        try:
            run_cell(
                config=config,
                policy=cell["policy"],
                server_overrides=cell["overrides"],
                variant_name=cell["variant"],
                concurrency=cell["concurrency"],
                repeats=repeats,
                workload_path=workload_path,
                run_dir=target,
                python=args.python,
                mini_sglang_dir=mini_sglang_dir,
                startup_timeout_s=args.startup_timeout_s,
                client_timeout_s=args.client_timeout_s,
            )
        except Exception as exc:  # keep sweeping; a failed cell is data too
            failures += 1
            print(f"    FAILED: {type(exc).__name__}: {exc}")
    print(json.dumps({"root": str(root), "failures": failures, "cells": len(cells)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
