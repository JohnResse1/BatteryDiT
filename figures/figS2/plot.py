"""figures/figS2 - cycle-level reconstruction-error distributions.

The 2x2 layout and violin/IQR encoding follow the legacy NC supplementary
figure.  Raw cycle-level values and the explicit plotting summaries are read
from ``analysis/outputs/``; this module does not read ``results/`` or compute
summary statistics.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from figures.common import export as export_helpers
from figures.common.style import MM, NC_WIDTH, apply_paper_style
from figures.common.tables import read_arrays


FIGURE_ID = "figS2"


TASKS = OrderedDict([
    ("downsample", "Downsampling"),
    ("random", "Random-point missing"),
    ("segment", "Segment missing"),
    ("destructive", "Destructive noise"),
])
MODELS = ("batterydit", "csdi", "transformer", "kalman")
MODEL_LABELS = OrderedDict([
    ("batterydit", "BatteryDiT"),
    ("csdi", "CSDI"),
    ("transformer", "Transformer"),
    ("kalman", "Kalman"),
])
MODEL_COLORS = {
    "batterydit": "#2166AC",
    "csdi": "#9FBBCF",
    "transformer": "#C76F52",
    "kalman": "#D4A08E",
}


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

    values = read_arrays("supplementary_waveform_rmse.npz")
    summary = read_arrays("supplementary_waveform_rmse_summary.npz")
    quantiles = np.asarray(summary["quantiles"], dtype=float)
    y_min = float(summary["y_min"][0])
    y_max = float(summary["y_max"][0])

    figure, axes = plt.subplots(
        2, 2, figsize=(NC_WIDTH, 125 * MM), sharey=True
    )
    figure.subplots_adjust(
        left=0.095, right=0.975, bottom=0.115, top=0.935,
        hspace=0.34, wspace=0.23,
    )

    positions = np.arange(1, len(MODELS) + 1)
    for panel_index, (condition, task_label) in enumerate(TASKS.items()):
        axis = axes.flat[panel_index]
        render_data = [values[f"{model}_{panel_index}"] for model in MODELS]
        parts = axis.violinplot(
            render_data,
            positions=positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.28,
            points=160,
        )

        for model, body in zip(MODELS, parts["bodies"]):
            color = MODEL_COLORS[model]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(0.65)
            body.set_alpha(0.38 if model != "batterydit" else 0.52)

        for model_index, (position, model) in enumerate(zip(positions, MODELS)):
            q1, median, q3 = quantiles[model_index, panel_index]
            color = MODEL_COLORS[model]
            axis.vlines(position, q1, q3, color=color, linewidth=3.2,
                        alpha=0.95, zorder=4)
            axis.hlines(median, position - 0.17, position + 0.17,
                        color="white", linewidth=1.15, zorder=5)
            axis.scatter(position, median, s=7, facecolor="white",
                         edgecolor=color, linewidth=0.55, zorder=6)

        axis.set_yscale("log")
        axis.set_ylim(y_min, y_max * 1.08)
        axis.set_xlim(0.48, len(MODELS) + 0.52)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [MODEL_LABELS[model] for model in MODELS],
            rotation=24, ha="right",
        )
        axis.set_title(task_label, fontsize=7.1, fontweight="bold", pad=3)
        style_axis(axis, grid_axis="y")

        if panel_index % 2 == 0:
            axis.set_ylabel("Cycle-level waveform RMSE (V)", labelpad=2)
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)
        add_panel_label(axis, f"({chr(97 + panel_index)})")

    previous_font_family = export_helpers.FONT_FAMILY
    export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
    plt.close(figure)
    return outputs
