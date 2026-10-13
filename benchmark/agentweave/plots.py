"""Paper figures for an AgentWeave report.

Deliberate deviations from the full chart spec, because the output is a static
PNG for a paper rather than an interactive HTML page:

* no hover layer or tooltip (a PNG has no pointer);
* no dark-mode variant, so one selected light palette is shipped instead of two;
* bar ends are square rather than 4px-rounded - matplotlib's ``bar`` has no
  rounded-cap primitive, and faking it with ``FancyBboxPatch`` breaks on stacked
  and negative segments.

Everything else is applied: one axis per panel (never two y-scales), categorical
hues assigned in fixed slot order, one legend per figure, direct value labels on
every non-trivial bar, a shared y-scale across small multiples, and a text table
shipped alongside (``report.json`` + ``format_table``) to satisfy the contrast
relief rule for the low-contrast series color.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Validated categorical slots 1-3 (blue / orange / aqua). Checked with the
# dataviz validator: adjacent-pair CVD and normal-vision gates pass in light
# mode. Aqua sits at 2.74:1 on the light surface, hence the relief rule - every
# bar is directly labeled and the report ships a text table.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"

_FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = _FONT
matplotlib.rcParams["axes.unicode_minus"] = False

# A ~2px surface gap between touching fills, per the mark spec.
_GAP = 1.5
# Segments/values smaller than this share of the axis are not labeled: the text
# would not fit on the mark.
_MIN_LABEL_SHARE = 0.06


def write_figures(
    report: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> list[Path]:
    """Render every figure that has data. Returns the paths actually written."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(report.get("rows", []))
    written: list[Path] = []
    for name, renderer in (
        ("fig_e2e_latency.png", _figure_e2e_latency),
        ("fig_continuation_gap.png", _figure_continuation_gap),
        ("fig_prewarm.png", _figure_prewarm),
        ("fig_kv_transfers.png", _figure_kv_transfers),
    ):
        figure = renderer(rows)
        if figure is None:
            continue
        path = out_dir / name
        figure.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(figure)
        written.append(path)
    return written


def _figure_e2e_latency(rows: Sequence[Mapping[str, Any]]) -> plt.Figure | None:
    return _latency_figure(
        rows,
        keys=("e2e_latency_ms_p50", "e2e_latency_ms_p95"),
        title="Agent end-to-end latency",
        ylabel="latency (ms)",
    )


def _figure_continuation_gap(rows: Sequence[Mapping[str, Any]]) -> plt.Figure | None:
    return _latency_figure(
        rows,
        keys=("continuation_first_token_ms_p50", "continuation_first_token_ms_p95"),
        title="Tool return to continuation first token",
        ylabel="latency (ms)",
    )


def _latency_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    title: str,
    ylabel: str,
) -> plt.Figure | None:
    """Grouped bars: columns are concurrencies, series are the percentiles.

    The x axis carries the dimension that varies besides concurrency - the KV
    experiment varies the policy, the prewarming experiment varies the server
    variant. When both vary the figure facets by variant (one row each) instead
    of putting every combination on one axis, which would leave nine
    unreadable labels per panel.

    All panels share one y-scale so the trend is readable across them.
    """

    concurrencies = _ordered({row["concurrency"] for row in rows})
    if not concurrencies:
        return None
    variants = _ordered({str(row["variant"]) for row in rows})
    has_variant = len(variants) > 1
    has_policy = len({str(row["policy"]) for row in rows if row["policy"]}) > 1
    # Facet by variant only when the policy also varies and would otherwise have
    # to share the x axis with it.
    row_labels: list[str | None] = variants if (has_variant and has_policy) else [None]

    def category_of(row: Mapping[str, Any]) -> str:
        return _short_policy(row["policy"]) if has_policy else str(row["variant"])

    labels = ("P50", "P95")
    figure, grid = plt.subplots(
        len(row_labels),
        len(concurrencies),
        figsize=(3.4 * len(concurrencies), 3.6 * len(row_labels)),
        facecolor=SURFACE,
        squeeze=False,
    )
    axes: list[plt.Axes] = []
    for row_index, row_label in enumerate(row_labels):
        for column_index, concurrency in enumerate(concurrencies):
            axis = grid[row_index][column_index]
            group = [
                row
                for row in rows
                if row["concurrency"] == concurrency
                and (row_label is None or str(row["variant"]) == row_label)
            ]
            if not _has_any(group, keys):
                axis.axis("off")
                continue
            ordered_group = sorted(group, key=category_of)
            _grouped_bars(
                axis,
                categories=[category_of(row) for row in ordered_group],
                series={
                    label: [row.get(key) for row in ordered_group]
                    for label, key in zip(labels, keys)
                },
                ylabel=ylabel if column_index == 0 else "",
            )
            caption = f"concurrency {concurrency}"
            if row_label is not None:
                caption = f"{row_label} - {caption}"
            axis.set_title(caption, color=INK_SECONDARY, fontsize=10, pad=22)
            axes.append(axis)
    if not axes:
        plt.close(figure)
        return None
    _share_ylimits(axes)
    _figure_legend(figure, axes[0])
    figure.suptitle(title, color=INK, fontsize=12)
    figure.tight_layout(rect=(0, 0.06 + 0.04 / len(row_labels), 1, 0.94))
    return figure


def _figure_prewarm(rows: Sequence[Mapping[str, Any]]) -> plt.Figure | None:
    """Sandbox acquisition mix (stacked) next to prediction accuracy."""

    variants = _ordered({str(row["variant"]) for row in rows})
    if not variants:
        return None
    sources = ("prewarm", "pool", "cold")
    has_sources = any(row.get(f"sandbox_{source}") for row in rows for source in sources)
    accuracy_rows = [row for row in rows if row.get("prediction_accuracy") is not None]
    if not has_sources and not accuracy_rows:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor=SURFACE, squeeze=False)
    left, right = axes[0][0], axes[0][1]

    if has_sources:
        _stacked_bars(
            left,
            categories=variants,
            series={
                source: [
                    _sum_for(rows, "variant", variant, f"sandbox_{source}") for variant in variants
                ]
                for source in sources
            },
            ylabel="sandbox acquisitions",
        )
        left.set_title("Sandbox acquisition source", color=INK_SECONDARY, fontsize=10, pad=22)
    else:
        left.axis("off")

    if accuracy_rows:
        # Keep the same x slots as the left panel so the variants line up; a
        # variant that made no prediction gets an explicit "n/a" instead of a
        # zero-length bar, which would read as "0% accurate".
        values: list[float | None] = []
        for variant in variants:
            group = [
                float(row["prediction_accuracy"])
                for row in accuracy_rows
                if str(row["variant"]) == variant
            ]
            values.append(sum(group) / len(group) if group else None)
        _grouped_bars(
            right,
            categories=variants,
            series={"accuracy": values},
            ylabel="prediction accuracy",
            percent=True,
            annotate_missing=True,
        )
        right.set_ylim(0, 1.05)
        right.set_title("Tool prediction accuracy", color=INK_SECONDARY, fontsize=10, pad=22)
    else:
        right.axis("off")

    for axis in (left, right):
        _axis_legend(axis)
    figure.suptitle("Speculative prewarming", color=INK, fontsize=12)
    figure.tight_layout(rect=(0, 0.05, 1, 0.90))
    return figure


def _figure_kv_transfers(rows: Sequence[Mapping[str, Any]]) -> plt.Figure | None:
    policies = _ordered({str(row["policy"]) for row in rows if row["policy"]})
    if not policies:
        return None
    has_pages = any(row.get("d2h_pages") or row.get("h2d_pages") for row in rows)
    has_actions = any(row.get("kv_actions") for row in rows)
    if not has_pages and not has_actions:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor=SURFACE, squeeze=False)
    left, right = axes[0][0], axes[0][1]
    categories = [_short_policy(policy) for policy in policies]

    if has_pages:
        _grouped_bars(
            left,
            categories=categories,
            series={
                "D2H (offload)": [
                    _sum_for(rows, "policy", policy, "d2h_pages") for policy in policies
                ],
                "H2D (restore)": [
                    _sum_for(rows, "policy", policy, "h2d_pages") for policy in policies
                ],
            },
            ylabel="KV pages",
        )
        left.set_title("Transferred KV pages", color=INK_SECONDARY, fontsize=10, pad=22)
    else:
        left.axis("off")

    if has_actions:
        action_names = _ordered({name for row in rows for name in (row.get("kv_actions") or {})})
        if action_names:
            _stacked_bars(
                right,
                categories=categories,
                series={
                    action: [
                        _counter_for(rows, "policy", policy, "kv_actions", action)
                        for policy in policies
                    ]
                    for action in action_names
                },
                ylabel="policy decisions",
            )
            right.set_title("KV policy actions", color=INK_SECONDARY, fontsize=10, pad=22)
        else:
            right.axis("off")
    else:
        right.axis("off")

    for axis in (left, right):
        _axis_legend(axis)
    figure.suptitle("KV tiering activity", color=INK, fontsize=12)
    figure.tight_layout(rect=(0, 0.05, 1, 0.90))
    return figure


def _grouped_bars(
    axis: plt.Axes,
    *,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float | None]],
    ylabel: str,
    percent: bool = False,
    annotate_missing: bool = False,
) -> None:
    labels = list(series)
    count = max(1, len(labels))
    width = 0.8 / count
    for index, label in enumerate(labels):
        values = list(series[label])
        offsets = [
            position + (index - (count - 1) / 2) * width for position in range(len(categories))
        ]
        heights = [None if value is None else float(value) for value in values]
        bars = axis.bar(
            offsets,
            [0.0 if height is None else height for height in heights],
            width=width,
            color=SERIES[index % len(SERIES)],
            edgecolor=SURFACE,
            linewidth=_GAP,
            label=label,
        )
        # A missing value gets no bar at all, so it cannot be misread as zero.
        for bar, height in zip(bars, heights):
            if height is None:
                bar.set_visible(False)
                if annotate_missing:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.0,
                        "n/a",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=MUTED,
                    )
        _label_bars(axis, bars, heights, percent=percent)
    _finalize(axis, categories, ylabel)


def _stacked_bars(
    axis: plt.Axes,
    *,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    ylabel: str,
) -> None:
    labels = list(series)
    bottoms = [0.0] * len(categories)
    for index, label in enumerate(labels):
        values = [float(value) for value in series[label]]
        axis.bar(
            range(len(categories)),
            values,
            bottom=bottoms,
            width=0.6,
            color=SERIES[index % len(SERIES)],
            edgecolor=SURFACE,
            linewidth=_GAP,
            label=label,
        )
        total = sum(values) or 1.0
        for position, value in enumerate(values):
            # Skip labels that would not fit inside their segment; the legend
            # and the text table still carry the value.
            if value / total < _MIN_LABEL_SHARE:
                continue
            axis.text(
                position,
                bottoms[position] + value / 2,
                _compact(value),
                ha="center",
                va="center",
                fontsize=7,
                color=SURFACE,
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    _finalize(axis, categories, ylabel)


def _label_bars(
    axis: plt.Axes,
    bars: Any,
    values: Sequence[float | None],
    *,
    percent: bool,
) -> None:
    """Direct labels so identity never rests on color alone."""

    positive = [value for value in values if value]
    ceiling = max(positive) if positive else 0.0
    for bar, value in zip(bars, values):
        if value is None or value == 0:
            continue
        if ceiling and value / ceiling < _MIN_LABEL_SHARE:
            continue
        text = f"{value * 100:.0f}%" if percent else _compact(value)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK_SECONDARY,
        )


def _finalize(axis: plt.Axes, categories: Sequence[str], ylabel: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_xticks(range(len(categories)))
    axis.set_xticklabels(categories, fontsize=8.5, color=INK_SECONDARY, rotation=30, ha="right")
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(AXIS)
    axis.tick_params(axis="y", colors=MUTED, labelsize=8)
    axis.tick_params(axis="x", length=0)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.xaxis.grid(False)
    axis.margins(x=0.06, y=0.18)


def _share_ylimits(axes: Sequence[plt.Axes]) -> None:
    """Give small multiples one y-scale so panels are comparable."""

    tops = [axis.get_ylim()[1] for axis in axes if axis.get_ylim()[1] > 0]
    if not tops:
        return
    top = max(tops)
    for axis in axes:
        axis.set_ylim(0, top)


def _figure_legend(figure: plt.Figure, axis: plt.Axes) -> None:
    """One shared legend for small multiples that share a series set."""

    handles, labels = axis.get_legend_handles_labels()
    if len(labels) < 2:
        return
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        bbox_to_anchor=(0.5, 0.0),
    )


def _axis_legend(axis: plt.Axes) -> None:
    """Per-panel legend above the axis.

    Placed per panel rather than per figure because the two panels of a
    comparison figure encode different series, and above rather than below
    because rotated x tick labels occupy the band under the axis.
    """

    handles, labels = axis.get_legend_handles_labels()
    if len(labels) < 2:
        return
    axis.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        handlelength=1.2,
        columnspacing=1.2,
    )


def _sum_for(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    value: Any,
    metric: str,
) -> float:
    return float(sum(int(row.get(metric) or 0) for row in rows if row.get(key) == value))


def _counter_for(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    value: Any,
    metric: str,
    action: str,
) -> float:
    return float(
        sum(int((row.get(metric) or {}).get(action, 0)) for row in rows if row.get(key) == value)
    )


def _has_any(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> bool:
    return any(row.get(key) is not None for row in rows for key in keys)


_POLICY_SHORT = {
    "no-control": "none",
    "tokencake": "tokencake",
    "continuum": "continuum",
}


def _short_policy(policy: Any) -> str:
    return _POLICY_SHORT.get(str(policy), str(policy))


def _ordered(values: Any) -> list[Any]:
    return sorted(values, key=lambda value: str(value))


def _compact(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"
