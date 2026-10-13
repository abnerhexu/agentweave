from __future__ import annotations

import json
from pathlib import Path

from benchmark.agentweave.report import (
    CONTINUATION_GAP_KEY,
    build_report,
    continuation_gaps_ms,
    dedupe_events,
    discover_runs,
    distribution,
    format_table,
    run_key,
    summarize_run,
)

MS = 1_000_000


def _event(lifecycle_id: str, event: str, timestamp_ms: float, **fields):
    return {
        "schema_version": 1,
        "event": event,
        "lifecycle_id": lifecycle_id,
        "timestamp_ns": int(timestamp_ms * MS),
        **fields,
    }


def test_run_key_groups_turns_of_one_agent():
    assert run_key("agent-0001:0") == "agent-0001"
    assert run_key("agent-0001:12") == "agent-0001"
    # A lifecycle with no turn suffix is its own run.
    assert run_key("agent-0001") == "agent-0001"


def test_continuation_gap_pairs_tool_finished_with_next_first_token():
    events = [
        _event("run-a:0", "request_received", 0.0),
        _event("run-a:0", "generation_first_token", 10.0),
        _event("run-a:0", "tool_finished", 100.0),
        _event("run-a:1", "continuation_submitted", 101.0),
        _event("run-a:1", "generation_first_token", 145.0),
        # Terminal turn: no continuation generation follows.
        _event("run-a:1", "tool_finished", 200.0),
    ]
    assert continuation_gaps_ms(events) == [45.0]


def test_continuation_gap_ignores_other_runs():
    events = [
        _event("run-a:0", "tool_finished", 0.0),
        _event("run-b:1", "generation_first_token", 10.0),
        _event("run-a:1", "generation_first_token", 30.0),
    ]
    assert continuation_gaps_ms(events) == [30.0]


def test_continuation_gap_uses_the_next_token_not_an_earlier_one():
    events = [
        _event("run-a:0", "generation_first_token", 0.0),
        _event("run-a:0", "tool_finished", 50.0),
        _event("run-a:1", "generation_first_token", 60.0),
    ]
    # The first token at t=0 precedes the tool, so it must not form a pair.
    assert continuation_gaps_ms(events) == [10.0]


def test_dedupe_keeps_the_earliest_timestamp_per_lifecycle_event():
    events = [
        _event("run-a:0", "tool_finished", 100.0, component="frontend"),
        _event("run-a:0", "tool_finished", 98.0, component="scheduler"),
    ]
    deduped = dedupe_events(events)
    assert len(deduped) == 1
    assert deduped[0]["component"] == "scheduler"


def test_dedupe_drops_records_without_a_lifecycle_or_timestamp():
    events = [{"event": "x"}, {"lifecycle_id": "run-a:0", "event": "y"}]
    assert dedupe_events(events) == []


def test_distribution_reports_percentiles():
    summary = distribution([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["count"] == 5
    assert summary["p50"] == 3.0
    assert summary["max"] == 100.0
    assert distribution([]) == {"count": 0}


def _write_run(root: Path, policy: str, concurrency: int, *, variant="default") -> Path:
    run_dir = root / policy / variant / f"c{concurrency}"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cell": {
                    "policy": policy,
                    "variant": variant,
                    "concurrency": concurrency,
                    "repeats": 1,
                },
                "config_name": "unit",
                "model": "test-model",
                "host_kv_pages": 4096 if policy == "tokencake" else 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "client.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "index": index,
                    "agent_id": f"agent-{index}",
                    "success": True,
                    "agent_status": "completed",
                    "latency_ms": 100.0 + index,
                    "steps": 2,
                    "tool_calls": 1,
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    events = [
        _event("agent-0:0", "request_received", 0.0),
        _event("agent-0:0", "generation_first_token", 20.0),
        _event("agent-0:0", "tool_started", 30.0),
        _event(
            "agent-0:0",
            "sandbox_acquired",
            35.0,
            detail={"source": "prewarm", "predicted_tool": "lookup", "correct_prediction": True},
        ),
        _event("agent-0:0", "tool_finished", 80.0),
        _event("agent-0:1", "generation_first_token", 90.0),
        _event(
            "agent-0:0",
            "kv_policy_decision",
            40.0,
            action="offload",
            implemented=True,
            copied_pages=4,
            reclaimed_pages=4,
            transfer_s=0.001,
        ),
    ]
    (run_dir / "events-scheduler-0.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    return run_dir


def test_summarize_run_joins_client_events_and_manifest(tmp_path):
    run_dir = _write_run(tmp_path, "tokencake", 4)
    summary = summarize_run(run_dir)

    assert summary["policy"] == "tokencake"
    assert summary["concurrency"] == 4
    assert summary["host_kv_pages"] == 4096
    assert summary["client"]["requests"] == 3
    assert summary["client"]["failures"] == 0
    assert summary["client"]["e2e_latency_ms"]["p50"] == 101.0
    assert summary[CONTINUATION_GAP_KEY]["p50"] == 10.0
    assert summary["timeline"]["sandbox_sources"] == {"prewarm": 1}
    assert summary["timeline"]["predictions"]["correct"] == 1
    assert summary["timeline"]["transfers"]["d2h_pages"] == 4
    assert summary["event_files"] == ["events-scheduler-0.jsonl"]


def test_discover_runs_skips_directories_without_client_results(tmp_path):
    _write_run(tmp_path, "no-control", 1)
    (tmp_path / "empty" / "c1").mkdir(parents=True)
    (tmp_path / "empty" / "c1" / "manifest.json").write_text("{}", encoding="utf-8")
    runs = discover_runs(tmp_path)
    assert [run["policy"] for run in runs] == ["no-control"]


def test_build_report_aggregates_per_variant_policy_and_concurrency(tmp_path):
    _write_run(tmp_path, "no-control", 1)
    _write_run(tmp_path, "tokencake", 1)
    report = build_report(discover_runs(tmp_path))

    rows = {(row["policy"], row["concurrency"]): row for row in report["rows"]}
    assert set(rows) == {("no-control", 1), ("tokencake", 1)}
    row = rows["tokencake", 1]
    assert row["requests"] == 3
    assert row["failures"] == 0
    assert row["continuation_first_token_ms_p50"] == 10.0
    assert row["prediction_accuracy"] == 1.0
    assert row["sandbox_prewarm"] == 1
    assert row["sandbox_cold"] == 0
    assert row["d2h_pages"] == 4
    assert row["kv_actions"] == {"offload": 1}
    assert report["runs"] == 2


def test_report_rows_separate_concurrencies(tmp_path):
    _write_run(tmp_path, "continuum", 1)
    _write_run(tmp_path, "continuum", 8)
    report = build_report(discover_runs(tmp_path))
    assert sorted(row["concurrency"] for row in report["rows"]) == [1, 8]


def test_table_view_renders_every_row(tmp_path):
    _write_run(tmp_path, "no-control", 1)
    report = build_report(discover_runs(tmp_path))
    table = format_table(report)
    assert "no-control" in table
    assert "cont_p50" in table
    assert len(table.splitlines()) == 3  # header, separator, one row
