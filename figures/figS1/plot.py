"""figures/figS1 - representative reconstruction comparison.

The layout and visual encoding follow the legacy NC supplementary
reconstruction panel.  The plotted arrays are the current analysis-layer
products, so the figure layer does not read ``results/`` or recompute metrics.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figures.common import export as export_helpers
from figures.common.style import NC_WIDTH, apply_paper_style
from figures.common.tables import read_arrays


FIGURE_ID = "figS1"


TASKS = OrderedDict([
    ("downsample", "Downsampling"),
    ("segment", "Segment missing"),
    ("random", "Random-point missing"),
    ("destructive", "Destructive noise"),
])
RECONSTRUCTION_MODELS = OrderedDict([
    ("batterydit", "BatteryDiT"),
    ("csdi", "CSDI"),
    ("transformer", "Transformer"),
    ("kalman", "Kalman"),
])

# Colours used by the legacy SI reconstruction comparison.  They are kept
# separate from the shared semantic palette because the old panel assigns a
# distinct muted colour to each reconstruction method.
MODEL_COLORS = {
    "batterydit": "#2166AC",
    "csdi": "#9FBBCF",
    "transformer": "#C76F52",
    "kalman": "#D4A08E",
}
C_REFERENCE = "#111111"
C_INPUT = "#999999"


def style_axis(axis):
    """Use the light grid and inward ticks of the legacy SI panel."""
    axis.tick_params(pad=1)
    axis.grid(True, axis="both", alpha=0.08, linewidth=0.3)
    axis.set_axisbelow(True)
    return axis


def draw(dataset: str = "mit", *, results_root=None):
    del dataset, results_root
    apply_paper_style()
    # The legacy NC exporter uses the installed Liberation Sans fallback for
    # Arial.  Select it explicitly here so the regenerated SI panel has the
    # same typography and does not emit font-resolution warnings.
    plt.rcParams.update({
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
    values = read_arrays("supplementary_representatives.npz")
    figure, axes = plt.subplots(
        4, 4, figsize=(NC_WIDTH, 165 / 25.4), sharex=True, sharey=True
    )
    figure.subplots_adjust(
        left=0.105, right=0.975, bottom=0.075, top=0.925,
        hspace=0.13, wspace=0.10,
    )

    for row, (condition, task_label) in enumerate(TASKS.items()):
        for column, (model, label) in enumerate(RECONSTRUCTION_MODELS.items()):
            axis = axes[row, column]
            clean = values[f"{model}_{condition}_clean"]
            observed = values[f"{model}_{condition}_observed"]
            reconstructed = values[f"{model}_{condition}_reconstructed"]
            steps = np.arange(len(clean))

            # Match the legacy corruption encoding: point/segment missingness
            # is shown as a low strip, while destructive values remain visible
            # but are clipped to the common voltage display range.
            if condition in ("random", "segment"):
                valid = observed > 1.8
                missing = ~valid
                axis.vlines(
                    steps[missing], 1.825, 1.865,
                    color=C_INPUT, linewidth=0.70, alpha=0.16, zorder=1,
                )
                axis.scatter(
                    steps[valid], observed[valid],
                    s=1.4, color=C_INPUT, alpha=0.48,
                    linewidths=0, zorder=2,
                )
            elif condition == "destructive":
                axis.plot(
                    steps, np.clip(observed, 1.86, 3.78),
                    color=C_INPUT, linewidth=0.75, alpha=0.48, zorder=1,
                )
            else:
                axis.plot(
                    steps, observed, color=C_INPUT,
                    linewidth=0.85, linestyle="--", alpha=0.65, zorder=1,
                )

            axis.plot(
                steps, reconstructed, color=MODEL_COLORS[model],
                linewidth=1.15, alpha=0.94, zorder=3,
            )
            axis.plot(
                steps, clean, color=C_REFERENCE,
                linewidth=0.80, alpha=0.92, zorder=4,
            )
            axis.set_xlim(0, len(steps) - 1)
            axis.set_ylim(1.82, 3.75)
            axis.set_yticks([2.0, 2.5, 3.0, 3.5])
            style_axis(axis)

            if column == 0:
                axis.text(
                    -0.24, 1.02, f"({chr(97 + row)})",
                    transform=axis.transAxes, fontsize=8,
                    fontweight="bold", va="bottom", ha="left",
                )
            if row == 0:
                axis.set_title(
                    label, fontsize=6.7, fontweight="bold",
                    color=MODEL_COLORS[model], pad=3,
                )
            if column == 0:
                axis.set_ylabel(
                    task_label, fontsize=6.6, fontweight="bold", labelpad=3,
                )
            if row == 3:
                axis.set_xlabel("Charging step", labelpad=1)

    legend_handles = [
        Line2D([0], [0], color=C_REFERENCE, linewidth=0.8, label="Ground truth"),
        Line2D([0], [0], color=C_INPUT, linewidth=0.8, linestyle="--", label="Observed"),
        Line2D([0], [0], color=MODEL_COLORS["batterydit"], linewidth=1.0,
               label="Reconstruction"),
    ]
    legend = axes[0, 0].legend(
        handles=legend_handles, loc="lower right", fontsize=5.5,
        handlelength=1.55, handletextpad=0.4, labelspacing=0.3,
        frameon=True, framealpha=0.82, borderpad=0.28,
    )
    legend.get_frame().set_linewidth(0.3)

    previous_font_family = export_helpers.FONT_FAMILY
    export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
    plt.close(figure)
    return outputs
