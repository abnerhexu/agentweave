from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from benchmark.agentweave import plots


def _row(**overrides):
    row = {
        "variant": "default",
        "policy": "no-control",
        "concurrency": 1,
        "requests": 10,
        "failures": 0,
        "e2e_latency_ms_p50": 100.0,
        "e2e_latency_ms_p95": 200.0,
        "continuation_first_token_ms_p50": 30.0,
        "continuation_first_token_ms_p95": 60.0,
        "prediction_accuracy": 0.9,
        "sandbox_prewarm": 5,
        "sandbox_pool": 2,
        "sandbox_cold": 3,
        "d2h_pages": 0,
        "h2d_pages": 0,
        "reclaimed_pages": 0,
        "failed_kv_actions": 0,
        "kv_actions": {"none": 10},
        "continuation_restore": {},
    }
    row.update(overrides)
    return row


def _bar_count(figure) -> int:
    return sum(len(axis.patches) for axis in figure.axes)


def test_latency_figure_facets_when_variant_and_policy_both_vary():
    """Regression: variants sharing a policy must not collapse into one bar."""

    variants = ("prewarm-off", "prewarm-confidence")
    rows = [
        _row(variant=variant, policy=policy)
        for variant in variants
        for policy in ("no-control", "tokencake", "continuum")
    ]
    figure = plots._figure_continuation_gap(rows)
    assert figure is not None
    # One row per variant, each carrying the three policies on its x axis.
    assert _bar_count(figure) == 6 * 2
    for axis in figure.axes:
        labels = {label.get_text() for label in axis.get_xticklabels()}
        assert labels == {"none", "tokencake", "continuum"}
    titles = {axis.get_title() for axis in figure.axes}
    assert titles == {
        "prewarm-off - concurrency 1",
        "prewarm-confidence - concurrency 1",
    }
    plt.close(figure)


def test_policy_only_report_labels_policies():
    rows = [_row(policy=policy) for policy in ("no-control", "tokencake", "continuum")]
    figure = plots._figure_e2e_latency(rows)
    assert figure is not None
    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert labels == ["continuum", "none", "tokencake"]
    plt.close(figure)


def test_small_multiples_share_one_y_scale():
    rows = [
        _row(concurrency=1, e2e_latency_ms_p50=10.0),
        _row(concurrency=8, e2e_latency_ms_p50=900.0),
    ]
    figure = plots._figure_e2e_latency(rows)
    assert figure is not None
    limits = {axis.get_ylim()[1] for axis in figure.axes}
    assert len(limits) == 1
    plt.close(figure)


def test_missing_accuracy_is_not_drawn_as_zero():
    rows = [
        _row(variant="prewarm-off", prediction_accuracy=None),
        _row(variant="prewarm-confidence", prediction_accuracy=0.75),
    ]
    figure = plots._figure_prewarm(rows)
    assert figure is not None
    right = figure.axes[1]
    visible = [bar for bar in right.patches if bar.get_visible()]
    assert len(visible) == 1
    texts = {text.get_text() for text in right.texts}
    assert "n/a" in texts
    plt.close(figure)


@pytest.mark.filterwarnings("ignore:More than 20 figures")
def test_write_figures_renders_every_figure_that_has_data(tmp_path):
    rows = [
        _row(variant=variant, policy=policy, concurrency=concurrency)
        for variant in ("prewarm-off", "prewarm-confidence-pool")
        for policy in ("no-control", "tokencake")
        for concurrency in (1, 8)
    ]
    written = plots.write_figures({"rows": rows}, rows, tmp_path)
    names = {path.name for path in written}
    assert names == {
        "fig_e2e_latency.png",
        "fig_continuation_gap.png",
        "fig_prewarm.png",
        "fig_kv_transfers.png",
    }
    for path in written:
        assert path.stat().st_size > 0


def test_write_figures_skips_figures_without_data(tmp_path):
    written = plots.write_figures({"rows": []}, [], tmp_path)
    assert written == []
