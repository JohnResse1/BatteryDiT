"""figures/figS4 - direct and reconstruction-assisted SOH prediction.

The 2x2 bar/violin layout follows the legacy NC supplementary benchmark
figure.  Current aggregate and grouped per-battery values are read from
``analysis/outputs/``; this module does not read ``results/`` or compute
summary statistics.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
import numpy as np

from figures.common import export as export_helpers
from figures.common.style import MM, NC_WIDTH, apply_paper_style
from figures.common.tables import read_arrays


FIGURE_ID = "figS4"


CONDITIONS = OrderedDict([
    ("clean", "Origin"),
    ("downsample", "Downsampling"),
    ("segment", "Segment missing"),
    ("random", "Random-point missing"),
    ("destructive", "Destructive noise"),
])
DIRECT_MODELS = OrderedDict([
    ("gru", ("GRU", "#C84F42")),
    ("lstm", ("LSTM", "#DD7058")),
    ("transformer", ("Transformer", "#E99178")),
    ("tcn", ("TCN", "#F0B291")),
    ("trb", ("TRB", "#AD7E6C")),
])
ASSISTED_MODELS = OrderedDict([
    ("batterydit", ("BatteryDiT+TRB", "#2166AC")),
    ("batterydit_tcn", ("BatteryDiT->TCN", "#3275B4")),
    ("kalman", ("Kalman+TRB", "#668EAC")),
    ("csdi", ("CSDI+TRB", "#8FB7CF")),
    ("transformer", ("Transformer+TRB", "#C1D2DC")),
])
GROUPS = (("direct", DIRECT_MODELS), ("assisted", ASSISTED_MODELS))


def style_axis(axis, grid_axis="both"):
    """Use the light grid and inward ticks of the legacy SI panel."""
    axis.tick_params(pad=1)
    axis.grid(True, axis=grid_axis, alpha=0.08, linewidth=0.3)
    axis.set_axisbelow(True)
    return axis


def add_panel_label(axis, label, x=-0.075, y=1.055):
    return axis.text(
        x, y, label, transform=axis.transAxes,
        fontsize=8, fontweight="bold", va="bottom", ha="left",
    )


def _apply_legacy_typography() -> None:
    """Match the effective NC typography used by the old SI exporter."""
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 6.0,
        "axes.labelsize": 6.7,
        "axes.titlesize": 7.2,
        "axes.titleweight": "bold",
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.1,
        "patch.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.10,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def draw(dataset: str = "mit", *, results_root=None):
    del dataset, results_root
    apply_paper_style()
    _apply_legacy_typography()

    values = read_arrays("supplementary_downstream.npz")
    summary = read_arrays("supplementary_downstream_summary.npz")
    condition_labels = list(CONDITIONS.values())
    condition_positions = np.arange(len(CONDITIONS))

    figure, axes = plt.subplots(
        2, 2, figsize=(NC_WIDTH, 158 * MM), sharey="row"
    )
    figure.subplots_adjust(
        left=0.085, right=0.975, bottom=0.085, top=0.955,
        hspace=0.34, wspace=0.20,
    )

    for group_index, (group, model_specs) in enumerate(GROUPS):
        bar_axis = axes[0, group_index]
        violin_axis = axes[1, group_index]
        models = list(model_specs)
        bar_width = 0.74 / len(models)

        # Top row: aggregate SOH RMSE by corruption condition.
        for model_index, (model, (label, color)) in enumerate(model_specs.items()):
            offset = (model_index - (len(models) - 1) / 2) * bar_width
            bar_axis.bar(
                condition_positions + offset,
                values[f"{group}_{model}_mean"],
                bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.84,
                label=label,
            )

        bar_axis.set_xticks(condition_positions)
        bar_axis.set_xticklabels(condition_labels, rotation=22, ha="right")
        bar_axis.set_ylim(0, float(summary["bar_ymax"][group_index]))
        style_axis(bar_axis, grid_axis="y")
        if group_index == 0:
            bar_axis.set_ylabel("SOH RMSE", labelpad=2)
        else:
            bar_axis.tick_params(axis="y", left=False, labelleft=False)

        legend_handles = [
            Patch(facecolor=color, edgecolor="none", alpha=0.84, label=label)
            for label, color in model_specs.values()
        ]
        bar_axis.legend(
            handles=legend_handles,
            loc="upper left",
            ncol=3,
            fontsize=5.5,
            handlelength=1.1,
            handletextpad=0.35,
            columnspacing=0.65,
            labelspacing=0.35,
        )

        # Bottom row: grouped per-battery SOH RMSE distributions.
        group_gap = 1.25
        group_width = len(models) + group_gap
        violin_data = []
        violin_positions = []
        violin_identity = []
        for condition_index, condition in enumerate(CONDITIONS):
            group_start = condition_index * group_width
            for model_index, model in enumerate(models):
                series = values[f"{group}_{model}_{condition}_battery_errors"]
                violin_data.append(series)
                violin_positions.append(group_start + model_index)
                violin_identity.append((model, condition_index))

        violin_parts = violin_axis.violinplot(
            violin_data,
            positions=violin_positions,
            widths=0.76,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.28,
            points=120,
        )
        for body, (model, _condition_index) in zip(
                violin_parts["bodies"], violin_identity):
            color = model_specs[model][1]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(0.45)
            body.set_alpha(0.46)

        for position, (model, condition_index) in zip(
                violin_positions, violin_identity):
            q1, median, q3 = summary[
                f"{group}_{model}_quantiles"
            ][condition_index]
            color = model_specs[model][1]
            violin_axis.vlines(
                position, q1, q3, color=color,
                linewidth=2.3, alpha=0.95, zorder=4,
            )
            violin_axis.hlines(
                median, position - 0.16, position + 0.16,
                color="white", linewidth=0.95, zorder=5,
            )

        violin_tick_positions = [
            condition_index * group_width + (len(models) - 1) / 2
            for condition_index in range(len(CONDITIONS))
        ]
        violin_axis.set_xticks(violin_tick_positions)
        violin_axis.set_xticklabels(condition_labels, rotation=22, ha="right")
        violin_axis.set_ylim(0, float(summary["violin_ymax"][group_index]))
        style_axis(violin_axis, grid_axis="y")
        if group_index == 0:
            violin_axis.set_ylabel("Battery-condition SOH RMSE", labelpad=2)
        else:
            violin_axis.tick_params(axis="y", left=False, labelleft=False)

    for panel_index, axis in enumerate(axes.flat):
        add_panel_label(axis, f"({chr(97 + panel_index)})")

    previous_font_family = export_helpers.FONT_FAMILY
    export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
    plt.close(figure)
    return outputs
