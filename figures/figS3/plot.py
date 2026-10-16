"""figures/figS3 - nominal SOH parity under reconstruction-assisted input.

The 2x2 layout, blue error encoding and shared colourbar follow the legacy NC
supplementary parity figure.  Current predictions and frozen plotting limits
are read from ``analysis/outputs/``; no source results or summary statistics
are computed in this module.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from figures.common import export as export_helpers
from figures.common.style import MM, NC_WIDTH, apply_paper_style
from figures.common.tables import read_arrays


FIGURE_ID = "figS3"


TASKS = OrderedDict([
    ("downsample", "Downsampling"),
    ("segment", "Segment missing"),
    ("random", "Random-point missing"),
    ("destructive", "Destructive noise"),
])
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "nominal_parity_error",
    ["#1A4778", "#3A6798", "#5A87B8", "#8AB0D8", "#BBD0E8"],
    N=256,
)
C_REFERENCE = "#D6604D"
SEED = 42


def style_axis(axis, grid_axis="both"):
    """Use the light grid and inward ticks of the legacy SI panel."""
    axis.tick_params(pad=1)
    axis.grid(True, axis=grid_axis, alpha=0.08, linewidth=0.3)
    axis.set_axisbelow(True)
    return axis


def add_panel_label(axis, label, x=-0.180, y=1.050):
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

    values = read_arrays("supplementary_soh_parity.npz")
    summary = read_arrays("supplementary_soh_parity_summary.npz")
    axis_min = float(summary["axis_min"][0])
    axis_max = float(summary["axis_max"][0])
    color_max = float(summary["color_max"][0])
    error_norm = Normalize(vmin=0.0, vmax=color_max, clip=True)

    figure, axes = plt.subplots(
        2, 2, figsize=(NC_WIDTH, 155 * MM), sharex=True, sharey=True
    )
    figure.subplots_adjust(
        left=0.100, right=0.850, bottom=0.100, top=0.930,
        hspace=0.25, wspace=0.30,
    )

    scatter = None
    for panel_index, (condition, condition_label) in enumerate(TASKS.items()):
        axis = axes.flat[panel_index]
        target = np.asarray(values[f"{condition}_target"], dtype=float)
        prediction = np.asarray(values[f"{condition}_prediction"], dtype=float)
        error = np.abs(prediction - target)

        # Match the legacy deterministic draw order so dense low-error points
        # do not systematically obscure the error-coloured tail.
        draw_order = np.random.default_rng(SEED + 500 + panel_index).permutation(len(target))
        scatter = axis.scatter(
            target[draw_order], prediction[draw_order],
            c=error[draw_order], s=1.5, cmap=ERROR_CMAP, norm=error_norm,
            alpha=0.50, edgecolors="none", zorder=2,
        )
        axis.plot(
            [axis_min, axis_max], [axis_min, axis_max],
            color=C_REFERENCE, linewidth=0.85, linestyle="-",
            alpha=0.75, zorder=5,
        )
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_box_aspect(1)
        axis.set_title(condition_label, fontsize=7.1, fontweight="bold", pad=3)
        style_axis(axis)

        if panel_index // 2 == 1:
            axis.set_xlabel("True nominal SOH", labelpad=1)
        if panel_index % 2 == 0:
            axis.set_ylabel("Predicted nominal SOH", labelpad=2)
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)
        add_panel_label(axis, f"({chr(97 + panel_index)})")

    colorbar_axis = figure.add_axes([0.895, 0.225, 0.018, 0.56])
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.set_label("Absolute SOH error", fontsize=5.7, labelpad=4)
    colorbar.ax.tick_params(labelsize=5.5, pad=1)
    colorbar.outline.set_linewidth(0.5)

    previous_font_family = export_helpers.FONT_FAMILY
    export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
    plt.close(figure)
    return outputs
