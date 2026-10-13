"""Summarize AgentWeave run directories into a comparison table and figures.

Each run directory produced by ``run_experiment`` holds a ``client.jsonl``
(per-agent end-to-end results), one ``events-scheduler-0.jsonl`` (the serving
timeline) and a ``manifest.json`` (the cell definition). This module joins them.

Besides the durations that ``minisgl.agentweave.analysis`` already reports, it
computes the headline metric of the proposal that spans two lifecycles:

    tool return  ->  first token of the continuation generation

The serving side records ``tool_finished`` on turn *t* and
``generation_first_token`` on turn *t+1*, each under its own ``lifecycle_id``
(``<run_id>:<turn>``). Neither module can compute the gap alone, so it is
reconstructed here by joining on the shared ``<run_id>`` prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.agentweave import ensure_mini_sglang_path

CONTINUATION_GAP_KEY = "continuation_first_token_ms"

# Run-directory layout: <root>/<policy>/<variant>/c<concurrency>/
_MANIFEST = "manifest.json"
_CLIENT = "client.jsonl"


def run_key(lifecycle_id: str) -> str:
    """Strip the ``:<turn>`` suffix so all turns of one agent run share a key."""

    head, separator, _tail = lifecycle_id.rpartition(":")
    return head if separator and head else lifecycle_id


def dedupe_events(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep the earliest timestamp per ``(lifecycle_id, event)``.

    The serving side can emit the same lifecycle event from more than one
    component. Collapsing to the earliest occurrence matches how
    ``summarize_events`` picks timestamps, and prevents duplicate pairings from
    inflating the continuation-gap samples.
    """

    best: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        lifecycle_id = record.get("lifecycle_id")
        event = record.get("event")
        timestamp = record.get("timestamp_ns")
        if not isinstance(lifecycle_id, str) or not isinstance(event, str):
            continue
        if not isinstance(timestamp, int):
            continue
        key = (lifecycle_id, event)
        current = best.get(key)
        if current is None or timestamp < current["timestamp_ns"]:
            best[key] = record
    return list(best.values())


def continuation_gaps_ms(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """For every ``tool_finished``, the gap to the next generation's first token.

    Pairs are matched within one agent run, so only the continuation that
    follows that tool call can complete the pair.
    """

    by_run: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for record in dedupe_events(records):
        by_run[run_key(str(record["lifecycle_id"]))].append(
            (int(record["timestamp_ns"]), str(record["event"]))
        )

    gaps: list[float] = []
    for timeline in by_run.values():
        timeline.sort()
        for index, (timestamp, event) in enumerate(timeline):
            if event != "tool_finished":
                continue
            for later_timestamp, later_event in timeline[index + 1 :]:
                if later_event == "generation_first_token":
                    gaps.append((later_timestamp - timestamp) / 1_000_000)
                    break
        # A tool_finished with no following generation is a terminal turn.
    return gaps


def load_client_results(path: str | Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            results.append(value)
    return results


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": fmean(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def percentile(ordered: Sequence[float], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Join client results, scheduler events and the manifest of one cell."""

    run_path = Path(run_dir)
    manifest = _read_json(run_path / _MANIFEST)
    cell = manifest.get("cell", {})

    results = load_client_results(run_path / _CLIENT)
    successes = [result for result in results if result.get("success")]
    incomplete = [
        result for result in successes if result.get("agent_status") not in (None, "completed")
    ]

    event_paths = sorted(run_path.glob("events-*.jsonl"))
    events: list[dict[str, Any]] = []
    for event_path in event_paths:
        events.extend(load_client_results(event_path))

    ensure_mini_sglang_path()
    from minisgl.agentweave.analysis import summarize_events

    timeline = summarize_events(events)
    gaps = continuation_gaps_ms(events)

    return {
        "run_dir": str(run_path),
        "policy": cell.get("policy"),
        "variant": cell.get("variant"),
        "concurrency": cell.get("concurrency"),
        "repeats": cell.get("repeats"),
        "config_name": manifest.get("config_name"),
        "host_kv_pages": manifest.get("host_kv_pages"),
        "model": manifest.get("model"),
        "submodule_sha": manifest.get("submodule_sha"),
        "repo_sha": manifest.get("repo_sha"),
        "client": {
            "requests": len(results),
            "failures": len(results) - len(successes),
            "incomplete": len(incomplete),
            "errors": sorted(
                Counter(
                    str(result.get("error")) for result in results if result.get("error")
                ).items(),
                key=lambda item: -item[1],
            )[:5],
            "e2e_latency_ms": distribution(
                [float(result["latency_ms"]) for result in successes if "latency_ms" in result]
            ),
            "tool_calls": sum(int(result.get("tool_calls") or 0) for result in successes),
            "steps": sum(int(result.get("steps") or 0) for result in successes),
        },
        "timeline": timeline,
        CONTINUATION_GAP_KEY: distribution(gaps),
        "event_files": [str(path.name) for path in event_paths],
    }


def discover_runs(root: str | Path) -> list[dict[str, Any]]:
    """Summarize every cell directory under ``root`` that has results."""

    root_path = Path(root)
    manifests = sorted(root_path.rglob(_MANIFEST))
    runs = []
    for manifest_path in manifests:
        run_path = manifest_path.parent
        if not (run_path / _CLIENT).exists():
            continue
        runs.append(summarize_run(run_path))
    return runs


def build_report(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group runs into a comparable table keyed by (variant, policy, concurrency)."""

    grouped: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run.get("variant"), run.get("policy"), run.get("concurrency"))].append(run)

    rows = []
    for (variant, policy, concurrency), group in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2] or 0)
    ):
        rows.append(
            {
                "variant": variant,
                "policy": policy,
                "concurrency": concurrency,
                "runs": len(group),
                "failures": sum(int(run["client"]["failures"]) for run in group),
                "requests": sum(int(run["client"]["requests"]) for run in group),
                "e2e_latency_ms_p50": _mean_of(group, ("client", "e2e_latency_ms", "p50")),
                "e2e_latency_ms_p95": _mean_of(group, ("client", "e2e_latency_ms", "p95")),
                "ttft_ms_p50": _mean_of(group, ("timeline", "durations", "ttft_ms", "p50")),
                "ttft_ms_p95": _mean_of(group, ("timeline", "durations", "ttft_ms", "p95")),
                "continuation_first_token_ms_p50": _mean_of(group, (CONTINUATION_GAP_KEY, "p50")),
                "continuation_first_token_ms_p95": _mean_of(group, (CONTINUATION_GAP_KEY, "p95")),
                "tool_window_ms_p50": _mean_of(
                    group, ("timeline", "durations", "tool_window_ms", "p50")
                ),
                "prediction_accuracy": _ratio_of_accuracy(group),
                "sandbox_prewarm": _sum_metric(group, ("timeline", "sandbox_sources", "prewarm")),
                "sandbox_pool": _sum_metric(group, ("timeline", "sandbox_sources", "pool")),
                "sandbox_cold": _sum_metric(group, ("timeline", "sandbox_sources", "cold")),
                "d2h_pages": _sum_metric(group, ("timeline", "transfers", "d2h_pages")),
                "h2d_pages": _sum_metric(group, ("timeline", "transfers", "h2d_pages")),
                "reclaimed_pages": _sum_metric(group, ("timeline", "transfers", "reclaimed_pages")),
                "failed_kv_actions": _sum_metric(
                    group, ("timeline", "transfers", "failed_actions")
                ),
                "kv_actions": _merge_counters(group, ("timeline", "kv_actions")),
                "continuation_restore": _merge_counters(
                    group, ("timeline", "continuation_restore")
                ),
            }
        )
    return {"rows": rows, "runs": len(runs)}


def _dig(source: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _mean_of(group: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values = [value for run in group if isinstance((value := _dig(run, path)), (int, float))]
    return fmean(values) if values else None


def _sum_metric(group: Sequence[Mapping[str, Any]], path: Sequence[str]) -> int:
    return sum(int(value) for run in group if isinstance((value := _dig(run, path)), int))


def _merge_counters(group: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for run in group:
        counters = _dig(run, path)
        if isinstance(counters, Mapping):
            for key, value in counters.items():
                if isinstance(value, int):
                    merged[str(key)] += value
    return dict(sorted(merged.items()))


def _ratio_of_accuracy(group: Sequence[Mapping[str, Any]]) -> float | None:
    correct = _sum_metric(group, ("timeline", "predictions", "correct"))
    incorrect = _sum_metric(group, ("timeline", "predictions", "incorrect"))
    total = correct + incorrect
    return correct / total if total else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    return value if isinstance(value, dict) else {}


def format_table(report: Mapping[str, Any]) -> str:
    """Plain-text table so the numbers are readable without the figures.

    This is also the required non-color view of the figures: the aqua series
    sits below 3:1 contrast on a light surface, so the values must be available
    as text.
    """

    columns = [
        ("variant", 20, "l"),
        ("policy", 11, "r"),
        ("conc", 5, "r"),
        ("req", 6, "r"),
        ("fail", 5, "r"),
        ("e2e_p50", 9, "r"),
        ("e2e_p95", 9, "r"),
        ("cont_p50", 9, "r"),
        ("cont_p95", 9, "r"),
        ("pre", 5, "r"),
        ("pool", 5, "r"),
        ("cold", 5, "r"),
        ("acc", 6, "r"),
        ("d2h", 7, "r"),
        ("h2d", 7, "r"),
    ]

    def cell(value: Any, width: int, align: str) -> str:
        if value is None:
            text = "-"
        elif isinstance(value, float):
            text = f"{value:.1f}"
        else:
            text = str(value)
        return text.ljust(width) if align == "l" else text.rjust(width)

    header = "".join(cell(name, width, align) for name, width, align in columns)
    lines = [header, "-" * len(header)]
    for row in report.get("rows", []):
        accuracy = row["prediction_accuracy"]
        values = [
            row["variant"] or "-",
            row["policy"] or "-",
            row["concurrency"],
            row["requests"],
            row["failures"],
            row["e2e_latency_ms_p50"],
            row["e2e_latency_ms_p95"],
            row["continuation_first_token_ms_p50"],
            row["continuation_first_token_ms_p95"],
            row["sandbox_prewarm"],
            row["sandbox_pool"],
            row["sandbox_cold"],
            None if accuracy is None else round(float(accuracy), 2),
            row["d2h_pages"],
            row["h2d_pages"],
        ]
        lines.append(
            "".join(
                cell(value, width, align) for value, (_name, width, align) in zip(values, columns)
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize AgentWeave run directories")
    parser.add_argument("root", help="Results root containing run directories")
    parser.add_argument("--output", help="Write the JSON report here (default: stdout only)")
    parser.add_argument("--runs-output", help="Also write the per-run summaries here")
    parser.add_argument("--plot-dir", help="Write PNG figures into this directory")
    parser.add_argument("--quiet", action="store_true", help="Do not print the table")
    args = parser.parse_args(argv)

    ensure_mini_sglang_path()
    runs = discover_runs(args.root)
    report = build_report(runs)

    if args.runs_output:
        Path(args.runs_output).write_text(
            json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not args.quiet:
        print(format_table(report))
    if args.plot_dir:
        from .plots import write_figures

        written = write_figures(report, runs, Path(args.plot_dir))
        print(json.dumps({"figures": [str(path) for path in written]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
