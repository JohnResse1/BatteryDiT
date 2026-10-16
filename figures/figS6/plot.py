"""Supplementary Fig. S6: battery-level Population and Individual recovery.

This is an R3 supporting figure. It deliberately shows only the two formal
reconstruction errors, E_pop and E_ind, as battery-level distributions under
progressive observation degradation. No directional-fidelity, amplitude,
phenotype, or auxiliary Individual-deviation taxonomy is introduced here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from figures.common import export as export_helpers
from figures.common.style import (
    CPLATE,
    MM,
    NC_WIDTH,
    apply_paper_style,
    panel_label,
)


FIGURE_ID = "figS6"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA_PATH = DATA / "s6_distribution.csv"

CONDITIONS = [
    ("downsample", "Downsampling", CPLATE[0]),
    ("random_missing", "Random-point missing", CPLATE[4]),
    ("segment_missing", "Segment missing", CPLATE[8]),
    ("destructive", "Destructive noise", CPLATE[12]),
    ("hybrid", "Hybrid", CPLATE[16]),
]
DIST_LEVELS = ("L1", "L3", "L5")


def _axis_style(axis: plt.Axes) -> None:
    axis.grid(True, axis="y", alpha=0.10, linewidth=0.35)
    axis.set_axisbelow(True)
    axis.tick_params(pad=1.2, length=2.2, width=0.55)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)


def _load() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    required = {
        "battery_id", "condition", "condition_label", "severity",
        "severity_index", "E_pop", "E_ind", "n_cycles",
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"S6 distribution asset is missing: {sorted(missing)}")
    if len(data) != 315 or data.battery_id.nunique() != 21:
        raise RuntimeError("S6 distribution asset must contain 21 batteries x 5 conditions x 3 severities")
    if set(data.condition) != {name for name, _label, _color in CONDITIONS}:
        raise RuntimeError("S6 distribution asset has an unexpected corruption condition")
    if set(data.severity) != set(DIST_LEVELS):
        raise RuntimeError("S6 distribution asset must contain L1, L3, and L5")
    if data.duplicated(["battery_id", "condition", "severity"]).any():
        raise RuntimeError("S6 distribution asset contains duplicate battery-condition-severity rows")
    if not np.isfinite(data[["E_pop", "E_ind"]].to_numpy(float)).all():
        raise RuntimeError("S6 E_pop/E_ind distribution asset contains non-finite values")
    return data.sort_values(["severity_index", "condition", "battery_id"], kind="stable")


def _draw_distribution(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    *,
    y_formatter: str,
    show_legend: bool,
) -> None:
    n_conditions = len(CONDITIONS)
    group_gap = 1.35
    group_width = n_conditions + group_gap
    positions = []
    identities = []
    distributions = []

    for severity_index, severity in enumerate(DIST_LEVELS):
        group_start = severity_index * group_width
        for condition_index, (condition, _label, _color) in enumerate(CONDITIONS):
            values = data.loc[
                data.condition.eq(condition) & data.severity.eq(severity), metric
            ].to_numpy(float)
            if len(values) != 21:
                raise RuntimeError(f"incomplete S6 coverage for {condition}/{severity}")
            positions.append(group_start + condition_index)
            identities.append(condition)
            distributions.append(values)

    parts = axis.violinplot(
        distributions,
        positions=positions,
        widths=0.76,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.28,
        points=120,
    )
    color_map = {condition: color for condition, _label, color in CONDITIONS}
    for body, condition in zip(parts["bodies"], identities):
        color = color_map[condition]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.45)
        body.set_alpha(0.46)

    for position, condition, values in zip(positions, identities, distributions):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        color = color_map[condition]
        axis.vlines(position, q1, q3, color=color, linewidth=2.3, alpha=0.95, zorder=4)
        axis.hlines(median, position - 0.16, position + 0.16,
                    color="white", linewidth=0.95, zorder=5)

    centers = [severity_index * group_width + (n_conditions - 1) / 2
               for severity_index in range(len(DIST_LEVELS))]
    axis.set_xticks(centers)
    axis.set_xticklabels(DIST_LEVELS)
    axis.set_xlabel("Corruption severity")
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=3.0, fontweight="bold")
    axis.set_xlim(-0.65, (len(DIST_LEVELS) - 1) * group_width + n_conditions - 0.35)
    # Reserve a clean upper band inside each panel for its own topology
    # legend, rather than placing the legend outside the plotting area.
    axis.set_ylim(0.0, float(np.nanmax(data[metric])) * 1.32)
    axis.yaxis.set_major_formatter(FormatStrFormatter(y_formatter))
    _axis_style(axis)

    if show_legend:
        handles = [
            Patch(facecolor=color, edgecolor=color, alpha=0.46, label=label)
            for _condition, label, color in CONDITIONS
        ]
        axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=3,
            fontsize=5.0,
            handlelength=0.85,
            handleheight=0.8,
            handletextpad=0.25,
            columnspacing=0.7,
            labelspacing=0.2,
            borderaxespad=0.0,
            frameon=False,
        )


def draw(dataset: str = "mit", *, results_root=None):
    del dataset, results_root
    apply_paper_style()
    data = _load()
    figure = plt.figure(figsize=(NC_WIDTH, 72 * MM), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        bottom=0.19,
        top=0.82,
        wspace=0.30,
    )
    axis_pop = figure.add_subplot(grid[0, 0])
    axis_ind = figure.add_subplot(grid[0, 1])

    _draw_distribution(
        axis_pop,
        data,
        "E_pop",
        r"Population error, $E_{\mathit{pop}}$",
        "Population recovery",
        y_formatter="%.2f",
        show_legend=True,
    )
    _draw_distribution(
        axis_ind,
        data,
        "E_ind",
        r"Individual error, $E_{\mathit{ind}}$",
        "Individual recovery",
        y_formatter="%.1f",
        show_legend=True,
    )

    panel_label(axis_pop, "a", x=-0.13, y=1.04)
    panel_label(axis_ind, "b", x=-0.13, y=1.04)

    outputs = export_helpers.export_figure(figure, FIGURE_ID)
    plt.close(figure)
    return outputs


if __name__ == "__main__":
    draw()
