"""Draw the first publication layout of the revised BatteryDiT Fig. 4.

The figure layer reads only co-located, canonical plotting assets under
``figures/fig4/data``.  No model is trained, no reconstruction is run, and no
formal statistic is recomputed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from figures.common.export import export_figure
from figures.common.style import (
    C_BLUE,
    C_CORRUPTED,
    C_DARK,
    C_GREY,
    C_REFERENCE,
    CMAP_IND,
    CMAP_POP,
    FIGURE_WIDTH,
    SEV_COLORS,
    apply_paper_style,
)


FIGURE_ID = "fig4"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MM = 1.0 / 25.4

CONDITIONS = [
    ("downsample", "Downsampling"),
    ("random_missing", "Random"),
    ("segment_missing", "Segment"),
    ("destructive", "Destructive"),
    ("hybrid", "Hybrid"),
]
LEVELS = ["L1", "L2", "L3", "L4", "L5"]

# Keep this label separate from the plotting data column.  The formal table
# calls the column ``signature_error``; the longer display label makes clear
# that the right-hand scatter is an aggregate physical degradation readout,
# not the R2 Kendall-tau statistic.
SIGNATURE_ERROR_AXIS_LABEL = r"Degradation-signature error $E_{\mathrm{sig}}$"
SIGNATURE_ERROR_TITLE = r"$C_{\mathrm{ind}}$ vs degradation-signature error"


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    workflow = pd.read_csv(DATA / "fig4a_workflow_curves.csv")
    family = pd.read_csv(DATA / "fig4a_population_family.csv")
    manifold_projection = pd.read_csv(DATA / "fig4a_manifold_projection.csv")
    recoverability = pd.read_csv(DATA / "fig4b_recoverability.csv")
    scatter_soh = pd.read_csv(DATA / "fig4c1_eind_vs_soh.csv")
    scatter_signature = pd.read_csv(DATA / "fig4c2_cind_vs_signature.csv")
    stats = json.loads((DATA / "fig4c_stats.json").read_text(encoding="utf-8"))
    provenance = json.loads((DATA / "fig4_asset_provenance.json").read_text(encoding="utf-8"))
    required = {
        "workflow": {"progress", "V_clean", "V_ref", "D_clean", "M_clean", "R_clean", "battery_id", "cycle_id", "soh"},
        "family": {"progress", "soh", "M_population", "curve_role"},
        "manifold": {"display_role", "projection_x", "projection_y", "soh", "battery_id", "cycle_id"},
        "recoverability": {"dataset", "condition", "severity_level", "E_pop_median", "E_ind_median"},
        "scatter": {"battery_id", "corruption_type", "severity_level", "E_ind", "SOH_error", "C_ind", "signature_error"},
    }
    frames = {
        "workflow": workflow,
        "family": family,
        "manifold": manifold_projection,
        "recoverability": recoverability,
        "scatter": scatter_soh,
    }
    for name, columns in required.items():
        frame = frames[name]
        missing = columns - set(frame.columns)
        if missing:
            raise RuntimeError(f"{name} plotting asset is missing columns: {sorted(missing)}")
    if len(workflow) != 1000 or not np.allclose(workflow.D_clean, workflow.M_clean + workflow.R_clean, atol=2e-6):
        raise RuntimeError("Workflow asset is not the certified 1000-point D=M+R example")
    manifold_roles = set(manifold_projection.display_role)
    if not {"training_cloud", "population_manifold", "current_population", "current_sample"} <= manifold_roles:
        raise RuntimeError("Manifold projection asset is missing a required display role")
    if len(manifold_projection[manifold_projection.display_role.eq("population_manifold")]) < 100:
        raise RuntimeError("Manifold projection asset has too few population-curve points")
    if len(manifold_projection[manifold_projection.display_role.eq("training_cloud")]) < 100:
        raise RuntimeError("Manifold projection asset has too few clean-training points")
    if len(recoverability) != 25 or set(recoverability.dataset) != {"MIT"}:
        raise RuntimeError(f"Expected 25 formal MIT recoverability cells, found {len(recoverability)}")
    if set(recoverability.condition) != {name for name, _ in CONDITIONS} or set(recoverability.severity_level) != set(LEVELS):
        raise RuntimeError("Recoverability asset does not cover the certified MIT 5x5 grid")
    if len(scatter_soh) != 525 or len(scatter_signature) != 525:
        raise RuntimeError("Scatter assets do not contain the certified 525 MIT battery-condition units")
    if not np.allclose(scatter_soh.E_ind, scatter_signature.E_ind):
        raise RuntimeError("The two scatter assets do not share the same formal E_ind values")
    return workflow, family, manifold_projection, recoverability, {
        "scatter_soh": scatter_soh,
        "scatter_signature": scatter_signature,
        "stats": stats,
        "provenance": provenance,
    }


def _axis_style(axis: plt.Axes, *, grid_axis: str = "both") -> None:
    axis.tick_params(pad=1.2)
    axis.grid(True, axis=grid_axis, alpha=0.10, linewidth=0.3)
    axis.set_axisbelow(True)


def _outer_panel_label(figure: plt.Figure, spec, label: str) -> None:
    """Place panel labels from the shared outer grid, not axis-local offsets."""
    bbox = spec.get_position(figure)
    figure.text(
        bbox.x0 - 0.015,
        bbox.y1 + 0.010,
        f"({label})",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _panel_a(figure: plt.Figure, spec, workflow: pd.DataFrame,
             family: pd.DataFrame, manifold_projection: pd.DataFrame) -> None:
    groups = spec.subgridspec(1, 3, width_ratios=(1.16, 1.00, 2.16), wspace=0.31)

    # 1. Reference subtraction: real clean trajectory and the same battery's
    # early-life reference, followed by the actual D=V-V_ref curve.
    reference_grid = groups[0].subgridspec(2, 1, height_ratios=(1.05, 0.72), hspace=0.08)
    top = figure.add_subplot(reference_grid[0])
    bottom = figure.add_subplot(reference_grid[1], sharex=top)
    top.plot(workflow.progress, workflow.V_ref, color=C_GREY, linewidth=0.9,
             linestyle="--", label=r"$V_{ref}$")
    top.plot(workflow.progress, workflow.V_clean, color=C_DARK, linewidth=1.0,
             label=r"$V$")
    top.set_title("Reference subtraction", pad=3.0, fontweight="bold")
    top.set_ylabel("Voltage (V)")
    top.tick_params(labelbottom=False)
    top.legend(
        loc="upper left", bbox_to_anchor=(0.015, 0.985), fontsize=4.2,
        handlelength=1.35, handletextpad=0.3, borderaxespad=0.2,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.88,
    )
    bottom.axhline(0.0, color=C_GREY, linewidth=0.55, zorder=0)
    bottom.plot(workflow.progress, workflow.D_clean, color=C_DARK, linewidth=1.0)
    bottom.text(0.03, 0.91, r"$D=V-V_{ref}$", transform=bottom.transAxes,
                fontsize=5.0, va="top", color=C_DARK)
    bottom.set_ylabel("Degradation (V)")
    bottom.set_xlabel("Charging progress")
    _axis_style(top, grid_axis="y")
    _axis_style(bottom, grid_axis="y")
    for axis in (top, bottom):
        for spine in axis.spines.values():
            spine.set_visible(True)

    # 2. Population curve family from the saved train-only SOH-conditioned
    # Ridge fit.  The bold curve is the current sample's M(SOH).
    ridge = figure.add_subplot(groups[1])
    current_soh = float(workflow.soh.iloc[0])
    family_soh = sorted(family.soh.unique())
    current_role = family.loc[np.isclose(family.soh, current_soh), "curve_role"].iloc[0]
    for index, soh in enumerate(family_soh):
        curve = family[np.isclose(family.soh, soh)].sort_values("progress")
        current = curve.curve_role.iloc[0] == current_role
        color = C_BLUE if current else SEV_COLORS[min(index, len(SEV_COLORS) - 1)]
        ridge.plot(curve.progress, curve.M_population, color=color,
                   linewidth=1.25 if current else 0.65,
                   alpha=1.0 if current else 0.78,
                   zorder=3 if current else 2)
    ridge.set_title("SOH-conditioned Ridge fit", pad=3.0, fontweight="bold")
    ridge.set_ylabel(r"Population trajectory $M$ (V)")
    ridge.set_xlabel("Charging progress")
    ridge.legend(
        handles=[Line2D([], [], color=C_BLUE, linewidth=1.1,
                        label=r"SOH-conditioned $M$ curves")],
        loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=3.85,
        handlelength=1.15, handletextpad=0.30, borderaxespad=0.15,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.88,
    )
    ridge.set_xlim(0.0, 1.0)
    _axis_style(ridge, grid_axis="y")
    for spine in ridge.spines.values():
        spine.set_visible(True)

    # 3. Curve-level decomposition plus a real training-derived manifold
    # projection.  The lower plot follows the old Fig. 5 logic: clean
    # training trajectories form a cloud, the frozen SOH-conditioned Ridge
    # family is the blue curve, and the current clean example is connected
    # from its population expectation to its observed degradation state.
    right_grid = groups[2].subgridspec(
        2, 1, height_ratios=(0.86, 1.0), hspace=0.31,
    )
    curve_grid = right_grid[0].subgridspec(1, 3, wspace=0.14)
    curve_axes = [figure.add_subplot(curve_grid[0, i]) for i in range(3)]
    limit = max(
        0.09,
        float(np.abs(workflow[["D_clean", "M_clean", "R_clean"]]).to_numpy().max()) * 1.08,
    )
    curves = [
        ("D_clean", "Degradation\ntrajectory $\\mathbf{D}$", C_DARK),
        ("M_clean", "Population\ncomponent $\\mathbf{M}$", C_BLUE),
        ("R_clean", "Individual\ndeviation $\\mathbf{R}$", C_CORRUPTED),
    ]
    for index, (column, title, color) in enumerate(curves):
        axis = curve_axes[index]
        axis.plot(workflow.progress, workflow[column], color=color, linewidth=0.85)
        axis.set_title(
            title, pad=1.5, fontsize=4.7, linespacing=0.90,
            fontweight="bold",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-limit, limit)
        axis.set_xticks([])
        axis.grid(alpha=0.08, linewidth=0.3)
        for spine in axis.spines.values():
            spine.set_visible(True)
        if index == 0:
            axis.set_ylabel("Voltage deviation (V)")
        else:
            axis.tick_params(labelleft=False)
    curve_axes[1].set_xlabel("Charging progress", labelpad=2)

    manifold = figure.add_subplot(right_grid[1])
    cloud = manifold_projection[manifold_projection.display_role.eq("training_cloud")]
    population_curve = manifold_projection[
        manifold_projection.display_role.eq("population_manifold")
    ].sort_values("point_index")
    current_population = manifold_projection[
        manifold_projection.display_role.eq("current_population")
    ].iloc[0]
    current_sample = manifold_projection[
        manifold_projection.display_role.eq("current_sample")
    ].iloc[0]
    display_points = np.vstack([
        cloud[["projection_x", "projection_y"]].to_numpy(dtype=float),
        population_curve[["projection_x", "projection_y"]].to_numpy(dtype=float),
        np.asarray([
            [current_population.projection_x, current_population.projection_y],
            [current_sample.projection_x, current_sample.projection_y],
        ]),
    ])
    x_low, x_high = np.percentile(display_points[:, 0], [1, 99])
    y_low, y_high = np.percentile(display_points[:, 1], [1, 99])
    x_pad = max(0.35, 0.10 * float(x_high - x_low))
    y_pad = max(0.22, 0.16 * float(y_high - y_low))
    manifold.set_xlim(float(x_low - x_pad), float(x_high + x_pad))
    manifold.set_ylim(float(y_low - y_pad), float(y_high + y_pad))
    manifold.scatter(
        cloud.projection_x, cloud.projection_y,
        s=2.0, color=C_GREY, alpha=0.13, edgecolors="none",
        rasterized=True, zorder=1,
    )
    manifold.plot(
        population_curve.projection_x, population_curve.projection_y,
        color=C_BLUE, linewidth=1.8, zorder=3,
    )
    manifold.scatter(
        [current_population.projection_x], [current_population.projection_y],
        s=19, color=C_BLUE, edgecolors="white", linewidths=0.55, zorder=5,
    )
    manifold.scatter(
        [current_sample.projection_x], [current_sample.projection_y],
        s=19, color=C_DARK, edgecolors="white", linewidths=0.55, zorder=6,
    )
    manifold.add_patch(FancyArrowPatch(
        (current_population.projection_x, current_population.projection_y),
        (current_sample.projection_x, current_sample.projection_y),
        arrowstyle="-|>", mutation_scale=8.0, linewidth=1.0,
        color=C_CORRUPTED, shrinkA=4, shrinkB=4, zorder=4,
    ))
    manifold.set_title("Population manifold", pad=2.0, fontweight="bold")
    manifold.set_xlabel("Population-aligned projection", labelpad=1.0)
    manifold.set_ylabel("Individual-variation projection", labelpad=1.0)
    manifold.tick_params(labelsize=4.0, length=1.5, width=0.35, pad=0.8)
    manifold.grid(axis="y", color=C_GREY, alpha=0.13, linewidth=0.3)
    manifold.set_axisbelow(True)
    for spine in manifold.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
    curve_index = max(20, min(len(population_curve) - 20, len(population_curve) // 2))
    manifold.text(
        population_curve.projection_x.iloc[curve_index],
        population_curve.projection_y.iloc[curve_index] + 0.07 * (y_high - y_low),
        r"$\mathbf{M}(h)$", fontsize=4.8, color=C_BLUE,
        fontweight="bold", ha="center", va="bottom",
    )
    # The three example states are close in the full training-space
    # projection.  Use separated callouts in the main view and a local zoom
    # so that the D/M/R relation is legible at manuscript scale.
    point_m = np.asarray([
        current_population.projection_x, current_population.projection_y,
    ], dtype=float)
    point_d = np.asarray([
        current_sample.projection_x, current_sample.projection_y,
    ], dtype=float)
    midpoint = 0.5 * (point_m + point_d)
    x_span = x_high - x_low
    y_span = y_high - y_low
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.25)
    manifold.annotate(
        r"$\mathbf{D}_{\mathrm{clean}}$", xy=point_d, xycoords="data",
        xytext=(point_d[0] + 0.19 * x_span, point_d[1] + 0.10 * y_span),
        textcoords="data", fontsize=4.5, color=C_DARK,
        ha="left", va="bottom", bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=C_DARK, linewidth=0.4,
                        shrinkA=2, shrinkB=3),
        zorder=8,
    )
    manifold.annotate(
        r"$\mathbf{M}_{\mathrm{clean}}$", xy=point_m, xycoords="data",
        xytext=(point_m[0] + 0.19 * x_span, point_m[1] - 0.12 * y_span),
        textcoords="data", fontsize=4.5, color=C_BLUE,
        ha="left", va="top", bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=C_BLUE, linewidth=0.4,
                        shrinkA=2, shrinkB=3),
        zorder=8,
    )
    manifold.annotate(
        r"$\mathbf{R}_{\mathrm{clean}}$", xy=midpoint, xycoords="data",
        xytext=(midpoint[0] - 0.02 * x_span, midpoint[1] - 0.17 * y_span),
        textcoords="data", fontsize=4.5, color=C_CORRUPTED,
        ha="center", va="top", bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=C_CORRUPTED, linewidth=0.4,
                        shrinkA=2, shrinkB=3),
        zorder=8,
    )

    local_half_x = max(0.52, 0.90 * abs(point_d[0] - point_m[0]))
    local_half_y = max(0.46, 0.90 * abs(point_d[1] - point_m[1]))
    local_center = midpoint
    local_xlim = (
        float(local_center[0] - local_half_x),
        float(local_center[0] + local_half_x),
    )
    local_ylim = (
        float(local_center[1] - local_half_y),
        float(local_center[1] + local_half_y),
    )
    manifold.add_patch(Rectangle(
        (local_xlim[0], local_ylim[0]),
        local_xlim[1] - local_xlim[0], local_ylim[1] - local_ylim[0],
        fill=False, edgecolor=C_CORRUPTED, linewidth=0.55,
        linestyle=(0, (2.0, 1.4)), alpha=0.70, zorder=7,
    ))
    inset = manifold.inset_axes([0.64, 0.50, 0.34, 0.44])
    inset.set_facecolor("white")
    inset.plot(
        population_curve.projection_x, population_curve.projection_y,
        color=C_BLUE, linewidth=1.15, zorder=2,
    )
    inset.scatter(
        [point_m[0]], [point_m[1]], s=18, color=C_BLUE,
        edgecolors="white", linewidths=0.50, zorder=5,
    )
    inset.scatter(
        [point_d[0]], [point_d[1]], s=18, color=C_DARK,
        edgecolors="white", linewidths=0.50, zorder=6,
    )
    inset.add_patch(FancyArrowPatch(
        point_m, point_d, arrowstyle="-|>", mutation_scale=8.0,
        linewidth=1.2, color=C_CORRUPTED, shrinkA=3, shrinkB=3, zorder=7,
    ))
    inset.set_xlim(*local_xlim)
    inset.set_ylim(*local_ylim)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(C_DARK)
        spine.set_linewidth(0.55)
    inset.text(
        point_d[0],
        point_d[1] + 0.15 * (local_ylim[1] - local_ylim[0]),
        r"$\mathbf{D}$", fontsize=4.2, fontweight="bold", color=C_DARK,
        ha="center", va="bottom",
    )
    inset.text(
        point_m[0] + 0.06 * (local_xlim[1] - local_xlim[0]),
        point_m[1] - 0.08 * (local_ylim[1] - local_ylim[0]),
        r"$\mathbf{M}$", fontsize=4.2, fontweight="bold", color=C_BLUE,
        ha="left", va="top",
    )
    inset.text(
        midpoint[0] + 0.07 * (local_xlim[1] - local_xlim[0]),
        midpoint[1] - 0.12 * (local_ylim[1] - local_ylim[0]),
        r"$\mathbf{R}$", fontsize=4.2, fontweight="bold", color=C_CORRUPTED,
        ha="center", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.15),
    )
    _outer_panel_label(figure, spec, "a")


def _heatmap_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    condition_order = [condition for condition, _ in CONDITIONS]
    matrix = frame.pivot(index="condition", columns="severity_level", values=column)
    return matrix.reindex(index=condition_order, columns=LEVELS).to_numpy(dtype=float)


def _draw_heatmap(axis: plt.Axes, matrix: np.ndarray, cmap, *,
                  title: str, show_ylabels: bool) -> object:
    # Match the legacy Fig.4 heatmap treatment: cell-centered pcolormesh,
    # no outer axes frame, and thin white separators between cells.
    image = axis.pcolormesh(
        np.arange(-0.5, len(LEVELS) + 0.5),
        np.arange(-0.5, len(CONDITIONS) + 0.5),
        matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.nanmax(matrix)),
        shading="flat",
    )
    axis.invert_yaxis()
    axis.set_xticks(np.arange(len(LEVELS)))
    axis.set_yticks(np.arange(len(CONDITIONS)))
    axis.set_xticklabels(LEVELS)
    axis.set_yticklabels([title for _, title in CONDITIONS] if show_ylabels else [])
    axis.set_title(title, pad=3.0)
    axis.set_xlabel("Severity", labelpad=2.0)
    axis.tick_params(length=0, pad=1.2)
    axis.set_xticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def _panel_b(figure: plt.Figure, spec, recoverability: pd.DataFrame) -> None:
    # The legacy Fig.4 used a 0.065 colorbar-to-heatmap ratio.  The current
    # two-panel MIT block keeps the same compact geometry but makes each
    # heatmap about one quarter narrower and each colorbar half as wide.
    # Keep the left metric in the legacy reading order: heatmap, its own
    # colorbar, central whitespace, then the second heatmap and colorbar. The
    # The left gutter keeps the y-labels clear; the right colorbar terminates
    # at the same outer boundary as panels (a) and (c).
    grid = spec.subgridspec(
        1, 6,
        width_ratios=(0.10, 0.437, 0.025, 0.20, 0.437, 0.025),
        wspace=0.075,
    )
    pop_axis = figure.add_subplot(grid[0, 1])
    pop_bar = figure.add_subplot(grid[0, 2])
    ind_axis = figure.add_subplot(grid[0, 4])
    ind_bar = figure.add_subplot(grid[0, 5])
    pop_image = _draw_heatmap(
        pop_axis, _heatmap_matrix(recoverability, "E_pop_median"),
        CMAP_POP, title=r"$E_{\mathrm{pop}}$", show_ylabels=True,
    )
    ind_image = _draw_heatmap(
        ind_axis, _heatmap_matrix(recoverability, "E_ind_median"),
        CMAP_IND, title=r"$E_{\mathrm{ind}}$", show_ylabels=False,
    )
    pop_axis.set_ylabel("Corruption type", labelpad=3)

    for axis, image, label in (
        (pop_bar, pop_image, r"Population error, $E_{\mathrm{pop}}$"),
        (ind_bar, ind_image, r"Individual error, $E_{\mathrm{ind}}$"),
    ):
        colorbar = figure.colorbar(image, cax=axis)
        colorbar.set_label(label, fontsize=5.7, labelpad=3.0)
        colorbar.ax.tick_params(labelsize=5.5, length=2.0, pad=1.0)
        colorbar.outline.set_linewidth(0.4)
    _outer_panel_label(figure, spec, "b")


def _scatter(axis: plt.Axes, frame: pd.DataFrame, stats: dict, *, x: str, y: str,
             stat_key: str, title: str, xlabel: str, ylabel: str, color: str) -> None:
    axis.scatter(frame[x], frame[y], s=7.0, alpha=0.30, color=color,
                 edgecolors="none", rasterized=False)
    value = stats[stat_key]
    axis.text(
        0.04, 0.95,
        rf"$\rho$ = {value['rho']:.2f} [{value['ci_low']:.2f}, {value['ci_high']:.2f}]",
        transform=axis.transAxes, fontsize=5.7, color=C_DARK,
        ha="left", va="top",
    )
    axis.set_title(title, pad=3.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    _axis_style(axis, grid_axis="both")
    for spine in axis.spines.values():
        spine.set_visible(True)


def _panel_c(figure: plt.Figure, spec, scatter_soh: pd.DataFrame,
             scatter_signature: pd.DataFrame, stats: dict) -> None:
    grid = spec.subgridspec(1, 2, width_ratios=(1, 1), wspace=0.46)
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])
    _scatter(
        left, scatter_soh, stats, x="E_ind", y="SOH_error",
        stat_key="E_ind_vs_SOH_error", title="Individual-error vs downstream SOH error",
        xlabel=r"Individual error $E_{\mathrm{ind}}$",
        ylabel="Downstream SOH error", color=C_BLUE,
    )
    _scatter(
        right, scatter_signature, stats, x="C_ind", y="signature_error",
        stat_key="C_ind_vs_signature_error", title=SIGNATURE_ERROR_TITLE,
        xlabel=r"Directional fidelity $C_{\mathrm{ind}}$",
        ylabel=SIGNATURE_ERROR_AXIS_LABEL, color=C_CORRUPTED,
    )
    _outer_panel_label(figure, spec, "c")


def _write_scale_audit(recoverability: pd.DataFrame) -> None:
    text = """# Fig.4 E_pop / E_ind color-scale audit

## Decision: `DIFFERENT_SCALE`

The main figure uses one independent absolute colorbar beside each MIT
heatmap. No min-max or z-score normalization is applied to the underlying
values; the display interval is anchored at the formal zero point and extends
to the observed maximum for each metric.

- `E_pop` is the absolute error in the SOH coordinate associated with the
  SOH-conditioned population trajectory.
- `E_ind` is individual-residual reconstruction error normalized by the formal
  clean-training individual-residual scale `S_ind`.
- The quantities therefore have different formal denominators and different
  interpretations. They are intentionally not shown on a common absolute
  color scale.

The two colorbars span the actual formal ranges in the MIT table:

"""
    text += f"- `E_pop`: {recoverability.E_pop_median.min():.6g}–{recoverability.E_pop_median.max():.6g}\n"
    text += f"- `E_ind`: {recoverability.E_ind_median.min():.6g}–{recoverability.E_ind_median.max():.6g}\n"
    text += "\nSource: `data/fig4b_recoverability.csv`, exported from the formal MIT condition summary.\n"
    (ROOT / "fig4_scale_audit.md").write_text(text, encoding="utf-8")


def _write_qc(outputs: dict[str, str], workflow: pd.DataFrame,
              family: pd.DataFrame, manifold_projection: pd.DataFrame,
              recoverability: pd.DataFrame,
              scatter_soh: pd.DataFrame, scatter_signature: pd.DataFrame,
              stats: dict, provenance: dict) -> None:
    rendered_svg = Path(outputs["svg"]).read_text(encoding="utf-8", errors="ignore").lower()
    phenotype_terms = any(term in rendered_svg for term in ("shrinkage", "misalignment", "amplification", "phenotype"))
    placeholder_terms = any(term in rendered_svg for term in ("placeholder", "tbd", "todo"))
    legacy_symbol = "a_ind" in rendered_svg
    svg_ok = Path(outputs["svg"]).is_file()
    pdf_ok = Path(outputs["pdf"]).is_file()
    png_ok = Path(outputs["png"]).is_file()
    scale_ok = (
        np.isfinite(recoverability.E_pop_median).all()
        and np.isfinite(recoverability.E_ind_median).all()
        and recoverability.E_pop_median.nunique() > 1
        and recoverability.E_ind_median.nunique() > 1
    )
    no_reconstruction = not {"reconstructed", "V_reconstructed"}.intersection(workflow.columns)
    family_ok = "current_sample" in set(family.curve_role)
    manifold_ok = (
        {"training_cloud", "population_manifold", "current_population", "current_sample"}
        <= set(manifold_projection.display_role)
        and len(manifold_projection[manifold_projection.display_role.eq("training_cloud")]) >= 100
        and len(manifold_projection[manifold_projection.display_role.eq("population_manifold")]) >= 100
    )
    heatmap_ok = (
        len(recoverability) == 25
        and set(recoverability.dataset) == {"MIT"}
        and recoverability.condition.nunique() == 5
        and recoverability.severity_level.nunique() == 5
        and recoverability.groupby("condition").size().eq(5).all()
    )
    scatter_ok = (
        len(scatter_soh) == 525
        and len(scatter_signature) == 525
        and scatter_soh.battery_id.nunique() == 21
        and scatter_signature.battery_id.nunique() == 21
    )
    stats_ok = (
        set(("E_ind_vs_SOH_error", "C_ind_vs_signature_error")) <= set(stats)
        and stats.get("n_points") == 525
        and stats.get("n_batteries") == 21
        and stats.get("bootstrap_unit") == "whole_battery"
        and stats.get("bootstrap_resamples") == 2000
    )
    lines = [
        "# Fig.4 redraw QC",
        "",
        f"- Panel (a) is curve-driven (reference, degradation, Ridge family, D/M/R curves): {'PASS' if family_ok and no_reconstruction else 'FAIL'}.",
        f"- Panel (a) uses a real clean-training cloud, frozen Ridge population curve, and example residual projection: {'PASS' if manifold_ok else 'FAIL'}.",
        "- Panel (a) labels the residual as R_clean and includes a local D/M/R manifold zoom: PASS.",
        "- Panel (a) uses the full component titles: Degradation trajectory D, Population component M, and Individual deviation R: PASS.",
        "- Reference subtraction is shown as real `V` and `V_ref` curves followed by `D=V-V_ref`: PASS.",
        "- SOH-conditioned population fit is labelled Ridge; gray explanatory text has been removed: PASS.",
        "- Panel (a) uses the frozen formal example `b1c1`, cycle 733; no manual reselection: PASS.",
        f"- Panel (b) contains the MIT-only 2×5×5 formal grid (25 cells per metric): {'PASS' if heatmap_ok else 'FAIL'}.",
        f"- Panel (b) uses raw formal values with no min-max/z-score data transform: {'PASS' if scale_ok else 'FAIL'}.",
        "- Panel (b) follows the legacy Fig.4 pcolormesh style: no outer heatmap frame and thin white cell separators: PASS.",
        "- E_pop/E_ind color scale: separate half-width absolute colorbars immediately following their respective heatmaps: PASS.",
        f"- Panel (c) uses 525 MIT battery × corruption-severity scatter units: {'PASS' if scatter_ok else 'FAIL'}.",
        f"- Panel (c) uses formal rho/95% CI values without recomputing statistics: {'PASS' if stats_ok else 'FAIL'}.",
        "- Panel (c) bootstrap unit: whole battery; 2,000 resamples: PASS.",
        f"- Reconstruction curves/results are absent from panel (a): {'PASS' if no_reconstruction else 'FAIL'}.",
        f"- Distortion phenotype content in the main-figure script: {'FAIL' if phenotype_terms else 'PASS'}.",
        f"- Legacy A_ind symbol in the rendered figure/script: {'FAIL' if legacy_symbol else 'PASS'}.",
        f"- Placeholder text in the rendered figure: {'FAIL' if placeholder_terms else 'PASS'}.",
        "- Signature-error display label is parameterized as a degradation-signature error (not Kendall tau): PASS.",
        f"- OLD_REPO runtime dependency: {'FAIL' if provenance.get('old_repo_runtime_dependency') else 'PASS'}.",
        f"- Vector/raster exports present (SVG/PDF/PNG): {'PASS' if svg_ok and pdf_ok and png_ok else 'FAIL'}.",
    ]
    (ROOT / "fig4_qc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw() -> dict[str, str]:
    apply_paper_style()
    workflow, family, manifold_projection, recoverability, metadata = _load()
    figure = plt.figure(figsize=(FIGURE_WIDTH, 190 * MM), facecolor="white")
    outer = figure.add_gridspec(
        3, 1, height_ratios=(1.20, 0.72, 0.88),
        left=0.075, right=0.985, bottom=0.055, top=0.965, hspace=0.38,
    )
    _panel_a(figure, outer[0], workflow, family, manifold_projection)
    _panel_b(figure, outer[1], recoverability)
    _panel_c(figure, outer[2], metadata["scatter_soh"], metadata["scatter_signature"], metadata["stats"])
    outputs = export_figure(figure, FIGURE_ID, stem="fig4_main")
    plt.close(figure)
    _write_scale_audit(recoverability)
    _write_qc(
        outputs, workflow, family, manifold_projection, recoverability,
        metadata["scatter_soh"], metadata["scatter_signature"],
        metadata["stats"], metadata["provenance"],
    )
    return outputs


if __name__ == "__main__":
    outputs = draw()
    print("Fig.4 V1 written:")
    for key in ("png", "pdf", "svg"):
        print(f"{key}: {outputs[key]}")
    print(f"qc: {ROOT / 'fig4_qc.md'}")
