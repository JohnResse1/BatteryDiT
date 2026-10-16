"""figures/figS5 - stress reconstruction trajectories.

The 5x5 trajectory layout follows the legacy NC Fig. S5/S8 panel.  All
clean, corrupted, and reconstructed curves are read from the analysis-layer
figure-ready archive; this module performs no inference or statistics.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np

from figures.common import export as export_helpers
from figures.common.style import MM, NC_WIDTH, apply_paper_style
from figures.common.tables import read_arrays


FIGURE_ID = "figS5"

TOPOLOGIES = OrderedDict([
    ("downsample", "Downsampling"),
    ("random_missing", "Random-point missing"),
    ("segment_missing", "Segment missing"),
    ("destructive", "Destructive noise"),
    ("hybrid", "Hybrid corruption"),
])
LEVELS = (1, 2, 3, 4, 5)
MISSING = np.asarray([0.10, 0.30, 0.50, 0.70, 0.90])
KEEP = 1.0 - MISSING
EPSILON = np.asarray([0.01, 0.03, 0.05, 0.10, 0.15])

C_CLEAN = "#111111"
C_OBSERVED = "#8B8B8B"
C_RECON = "#2166AC"
C_MISSING = "#D8D8D8"


def _apply_legacy_typography() -> None:
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


def _style_axis(axis) -> None:
    axis.tick_params(pad=1)
    axis.grid(True, alpha=0.08, linewidth=0.3)
    axis.set_axisbelow(True)


def _severity_text(topology: str, level_index: int) -> str:
    if topology == "downsample":
        return f"{KEEP[level_index] * 100:.0f}% retained"
    if topology in {"random_missing", "segment_missing"}:
        return f"{MISSING[level_index] * 100:.0f}% missing"
    if topology == "hybrid":
        return rf"{MISSING[level_index] * 100:.0f}% miss.; $\epsilon$={EPSILON[level_index]:.2f} V"
    return rf"$\epsilon$ = {EPSILON[level_index]:.2f} V"


def _missing_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def draw(dataset: str = "mit", *, results_root=None):
    del dataset, results_root
    apply_paper_style()
    _apply_legacy_typography()
    values = read_arrays("stress_test_representative_trajectories.npz")

    clean = np.asarray(values["clean"], dtype=float).reshape(-1)
    progress = np.linspace(0.0, 1.0, len(clean))
    figure, axes = plt.subplots(
        5, 5, figsize=(NC_WIDTH, 212 * MM), sharex=True, sharey=True,
    )
    figure.subplots_adjust(
        left=0.085, right=0.988, bottom=0.082, top=0.930,
        hspace=0.19, wspace=0.10,
    )
    for column, level in enumerate(LEVELS):
        axes[0, column].set_title(f"L{level}", fontweight="bold", pad=3)

    for row, (topology, topology_label) in enumerate(TOPOLOGIES.items()):
        for column, level in enumerate(LEVELS):
            axis = axes[row, column]
            key = f"{topology}_L{level}"
            corrupted = np.asarray(values[f"{key}_corrupted"], dtype=float).reshape(-1)
            reconstructed = np.asarray(values[f"{key}_reconstructed"], dtype=float).reshape(-1)
            observed = (
                np.abs(corrupted) > 1e-12
                if topology != "destructive"
                else np.ones_like(corrupted, dtype=bool)
            )

            if topology in {"downsample", "random_missing", "hybrid"}:
                axis.scatter(
                    progress[observed], corrupted[observed], s=1.5,
                    color=C_OBSERVED, alpha=0.52, edgecolors="none", zorder=2,
                )
            elif topology == "segment_missing":
                shown = corrupted.copy()
                shown[~observed] = np.nan
                axis.plot(progress, shown, color=C_OBSERVED, lw=0.65,
                          alpha=0.72, zorder=2)
                for start, end in _missing_runs(~observed):
                    axis.axvspan(
                        progress[start], progress[end], color=C_MISSING,
                        alpha=0.32, lw=0, zorder=0,
                    )
            else:
                axis.plot(progress, corrupted, color=C_OBSERVED, lw=0.45,
                          alpha=0.50, zorder=1)

            axis.plot(progress, clean, color=C_CLEAN, lw=0.75,
                      alpha=0.95, zorder=4)
            axis.plot(progress, reconstructed, color=C_RECON, lw=1.05,
                      zorder=3)
            axis.text(
                0.025, 0.955, _severity_text(topology, level - 1),
                transform=axis.transAxes, ha="left", va="top",
                fontsize=4.8, color="#444444",
            )
            rmse = float(np.asarray(values[f"{key}_rmse"]).reshape(-1)[0])
            axis.text(
                0.975, 0.055, f"RMSE {rmse:.3f} V",
                transform=axis.transAxes, ha="right", va="bottom",
                fontsize=4.6, color=C_RECON,
            )
            if row == 0 and column == 0:
                axis.legend(
                    handles=[
                        Line2D([], [], color=C_CLEAN, lw=0.9, label="Clean"),
                        Line2D([], [], color=C_OBSERVED, lw=0.8, label="Corrupted"),
                        Line2D([], [], color=C_RECON, lw=1.2, label="BatteryDiT"),
                    ],
                    loc="lower right", bbox_to_anchor=(0.985, 0.12),
                    borderaxespad=0, handlelength=1.35, handletextpad=0.45,
                    labelspacing=0.18, fontsize=4.2, frameon=False,
                )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(1.95, 3.78)
            _style_axis(axis)
            if column == 0:
                axis.set_ylabel(f"{topology_label}\nVoltage (V)", labelpad=3)
            if row == len(TOPOLOGIES) - 1:
                axis.set_xlabel("Normalized charging progress", labelpad=1)

    for row, panel_label in enumerate(("(a)", "(b)", "(c)", "(d)", "(e)")):
        axes[row, 0].text(
            -0.25, 1.04, panel_label, transform=axes[row, 0].transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left",
        )

    previous_font_family = export_helpers.FONT_FAMILY
    export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
    plt.close(figure)
    return outputs
