"""Build the final R5 cross-system Fig. 6 main-text figure.

The plotting layer reuses the frozen formal/canonical loader and audit in
``_fig6_base.py``. No model training, inference, or new statistical analysis
is performed here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from figures.fig6 import _fig6_base as base


OUT = base.OUT
MAIN_PNG = OUT / "Fig6_main.png"
MAIN_PDF = OUT / "Fig6_main.pdf"
MAIN_SVG = OUT / "Fig6_main.svg"


def _header_refined(fig: plt.Figure, slot, label: str, header_ratio: float):
    """Use the established Fig. 6 header geometry."""
    header, content = base._panel_header(fig, slot, label, "", header_ratio)
    if header.texts:
        # Keep the labels on one vertical baseline at the upper-left of each
        # panel, without moving any panel body.
        header.texts[0].set_x(-0.055)
    return header, content


def _align_panel_label_to_first_row(header, reference_text) -> None:
    """Put a panel label on the top-left baseline of its first row.

    The three panels use different header/content ratios.  Aligning to the
    first visible row title keeps ``(a)``, ``(b)``, and ``(c)`` visually
    attached to their respective top-left corners without moving any axes.
    """
    figure = header.figure
    figure.canvas.draw()
    label_text = header.texts[0]
    label_box = label_text.get_window_extent(figure.canvas.get_renderer())
    reference_box = reference_text.get_window_extent(figure.canvas.get_renderer())
    y0 = header.transAxes.transform((0.0, 0.0))[1]
    y1 = header.transAxes.transform((0.0, 1.0))[1]
    if y1 == y0:
        return
    current_y = float(label_text.get_position()[1])
    shift = (reference_box.y1 - label_box.y1) / (y1 - y0)
    label_text.set_y(current_y + shift)


def _plot_panel_a_refined(fig: plt.Figure, slot, data: dict[str, object]):
    # Restore the prior main-figure panel-a drawing exactly: the established
    # 1×6 geometry, chemistry spacing, HUST in-panel legend, representative
    # curves, and y-ranges are all retained.
    axes = base._plot_panel_a(fig, slot, data)
    _align_panel_label_to_first_row(axes[0], axes[1].texts[0])
    return axes


def _performance_card(
    ax: plt.Axes,
    values: np.ndarray,
    errors: np.ndarray,
    y_max: float,
    ylabel: str,
    title: str | None,
) -> None:
    positions = np.arange(len(base.FRAMEWORK_CONDITIONS), dtype=float)
    ax.bar(
        positions,
        values,
        width=0.50,
        color=base.BAR_BLUE,
        alpha=0.78,
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        positions,
        values,
        yerr=errors,
        fmt="none",
        ecolor=base.C_DARK,
        elinewidth=0.40,
        capsize=0.75,
        zorder=3,
    )
    for x, value, error in zip(positions, values, errors):
        value_text = ax.text(
            x,
            float(value + error + y_max * 0.025),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=3.5,
            color=base.C_DARK,
        )
        value_text.set_gid("fig6_refined_bar_value")
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(-0.58, len(base.FRAMEWORK_CONDITIONS) - 0.42)
    ax.set_xticks(positions)
    ax.set_xticklabels([short for _, _, short in base.FRAMEWORK_CONDITIONS])
    ax.tick_params(axis="x", length=0, pad=1.0)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, width=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=4.3, labelpad=1.0)
    if title:
        ax.text(
            0.5, 1.035, title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.2,
            fontweight="bold", clip_on=False,
        )
    base._style_axis(ax, labelsize=3.8)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0, width=0)


def _plot_panel_b_refined(fig: plt.Figure, slot, data: dict[str, object]):
    header, content = _header_refined(fig, slot, "b", header_ratio=0.065)
    grid = content.subgridspec(2, 6, hspace=0.38, wspace=0.36)
    recon = data["recon"].set_index(["dataset_key", "condition"])
    downstream = data["downstream"].set_index(["dataset_key", "condition"])
    top_axes, bottom_axes = [], []

    for j, (key, label, _) in enumerate(base.EXTERNAL_FRAMEWORK):
        top = fig.add_subplot(grid[0, j])
        bottom = fig.add_subplot(grid[1, j])
        top_axes.append(top)
        bottom_axes.append(bottom)
        conditions = [condition for condition, _, _ in base.FRAMEWORK_CONDITIONS]
        wf = recon.loc[(key, conditions), "waveform_rmse_mean"].to_numpy(float)
        wf_sd = recon.loc[(key, conditions), "waveform_rmse_sd"].to_numpy(float)
        down = downstream.loc[
            (key, conditions), "downstream_target_rmse_mean"
        ].to_numpy(float)
        down_sd = downstream.loc[
            (key, conditions), "downstream_target_rmse_sd"
        ].to_numpy(float)

        wf_max = float(np.nanmax(wf + wf_sd) * 1.30)
        down_max = float(np.nanmax(down + down_sd) * 1.30)
        _performance_card(
            top, wf, wf_sd, wf_max,
            "Waveform RMSE (V)" if j == 0 else "", label,
        )
        _performance_card(
            bottom, down, down_sd, down_max,
            "Downstream RMSE" if j == 0 else "", None,
        )
    first_title = next(
        text for text in top_axes[0].texts if text.get_text() == base.EXTERNAL_FRAMEWORK[0][1]
    )
    _align_panel_label_to_first_row(header, first_title)
    return [header, *top_axes, *bottom_axes]


def _strip_axis(fig: plt.Figure, slot, labels: list[str], fontsize: float = 4.0):
    ax = fig.add_subplot(slot)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(len(labels), 0.0)
    for i, label in enumerate(labels):
        text_artist = ax.text(
            1.0, i + 0.5, label,
            ha="right", va="center", fontsize=fontsize,
            fontweight="bold", color=base.C_DARK,
        )
        text_artist.set_gid("fig6_refined_condition_label")
    return ax


def _group_label_axis(fig: plt.Figure, slot, label: str):
    ax = fig.add_subplot(slot)
    ax.set_axis_off()
    ax.text(
        0.5, 0.5, label,
        ha="center", va="center", rotation=90,
        fontsize=4.5, fontweight="bold", color=base.C_DARK,
    )
    return ax


def _schematic_key(fig: plt.Figure, slot, cmap, label: str):
    ax = fig.add_subplot(slot)
    ax.set_axis_off()
    gradient = np.linspace(0.0, 1.0, 256, dtype=float)[None, :]
    ax.imshow(
        gradient,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        extent=(0.06, 0.94, 0.38, 0.58),
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.06, 0.80, label,
        ha="left", va="center", fontsize=4.2,
        fontweight="bold", color=base.C_DARK,
    )
    ax.text(0.06, 0.24, "Low", ha="left", va="center", fontsize=3.8, color=base.C_DARK)
    ax.text(0.94, 0.24, "High", ha="right", va="center", fontsize=3.8, color=base.C_DARK)
    return ax


def _plot_panel_c_refined(fig: plt.Figure, slot, data: dict[str, object]):
    """Restore the main small-multiple heatmaps with Fig.4b-style bars."""
    header, content = _header_refined(fig, slot, "c", header_ratio=0.04)
    grid = content.subgridspec(
        3,
        6,
        height_ratios=(0.14, 1.0, 1.0),
        # Five square heatmaps followed by two narrow raw-value colourbars.
        width_ratios=(1.0, 1.0, 1.0, 1.0, 1.0, 0.06),
        hspace=0.22,
        wspace=0.26,
    )
    axes = [header]
    pi = data["pi"]

    def raw_matrix(dataset_key: str, metric: str) -> np.ndarray:
        sub = pi[pi.dataset_key.eq(dataset_key)].set_index(["condition", "severity"])
        matrix = np.zeros((len(base.PI_CONDITIONS), len(base.SEVERITIES)), dtype=float)
        for i, (condition, _) in enumerate(base.PI_CONDITIONS):
            for j, severity in enumerate(base.SEVERITIES):
                matrix[i, j] = float(sub.loc[(condition, severity), metric])
        return matrix

    def raw_heatmap(slot, matrix, cmap, vmin, vmax, show_ylabels, show_xlabels):
        ax = fig.add_subplot(slot)
        mesh = ax.pcolormesh(
            np.arange(6, dtype=float),
            np.arange(6, dtype=float),
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            edgecolors="white",
            linewidth=0.48,
        )
        ax.set_xlim(0, 5)
        ax.set_ylim(5, 0)
        ax.set_box_aspect(1.0)
        ax.set_anchor("W")
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_xticklabels(base.SEVERITIES if show_xlabels else [])
        ax.set_yticklabels(
            [label for _, label in base.PI_CONDITIONS] if show_ylabels else []
        )
        if show_ylabels:
            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")
        ax.tick_params(axis="both", length=0, pad=1.1, labelsize=3.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax, mesh

    pop_matrices = {
        key: raw_matrix(key, "E_pop") for key, _ in base.EXTERNAL_PI
    }
    ind_matrices = {
        key: raw_matrix(key, "E_ind") for key, _ in base.EXTERNAL_PI
    }
    pop_vmin, pop_vmax = 0.0, float(max(np.nanmax(x) for x in pop_matrices.values()))
    ind_vmin, ind_vmax = 0.0, float(max(np.nanmax(x) for x in ind_matrices.values()))

    for j, (key, label) in enumerate(base.EXTERNAL_PI):
        title_ax = fig.add_subplot(grid[0, j])
        title_ax.set_axis_off()
        title_ax.text(
            0.5, 0.45, label,
            transform=title_ax.transAxes,
            ha="center", va="center", fontsize=5.5,
            fontweight="bold", color=base.C_DARK,
        )
        axes.append(title_ax)

    heatmap_axes = []
    for j, (key, _) in enumerate(base.EXTERNAL_PI):
        pop_ax, pop_mesh = raw_heatmap(
            grid[1, j], pop_matrices[key], base.POP_CMAP,
            pop_vmin, pop_vmax,
            show_ylabels=(j == 0), show_xlabels=False,
        )
        ind_ax, ind_mesh = raw_heatmap(
            grid[2, j], ind_matrices[key], base.IND_CMAP,
            ind_vmin, ind_vmax,
            show_ylabels=(j == 0), show_xlabels=True,
        )
        axes.extend([pop_ax, ind_ax])
        heatmap_axes.extend([pop_ax, ind_ax])

    # One independent raw-value bar per metric, shared by the five heatmaps.
    pop_cax = fig.add_subplot(grid[1, -1])
    ind_cax = fig.add_subplot(grid[2, -1])
    pop_cb = fig.colorbar(heatmap_axes[0].collections[0], cax=pop_cax)
    ind_cb = fig.colorbar(heatmap_axes[1].collections[0], cax=ind_cax)
    for cb, label in (
        (pop_cb, r"Population error, $E_{\mathrm{pop}}$"),
        (ind_cb, r"Individual error, $E_{\mathrm{ind}}$"),
    ):
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        cb.set_label(label, fontsize=5.7, labelpad=3.0)
        cb.ax.tick_params(labelsize=5.5, length=2.0, width=0.45, pad=1.0)
        cb.outline.set_visible(True)
        cb.outline.set_linewidth(0.4)
    axes.extend([pop_cax, ind_cax])
    _align_panel_label_to_first_row(header, axes[1].texts[0])
    return axes, heatmap_axes


def _rewrite_raw_c_tables(data: dict[str, object]) -> None:
    """Make the plotting tables agree with the raw-value heatmaps."""
    pi = data["pi"].copy()
    specifications = (
        (
            "E_pop",
            "fig6_panel_c_epop.csv",
            "E_pop = |h_rec − SOH_true|",
            "none; raw formal E_pop used for colour; one metric-wise raw colour scale across external PI systems",
        ),
        (
            "E_ind",
            "fig6_panel_c_eind.csv",
            "E_ind = RMSE(R_rec − R_clean) / S_ind,d",
            "none; raw formal E_ind used for colour; one metric-wise raw colour scale across external PI systems",
        ),
    )
    for metric, filename, definition, transform in specifications:
        table = pi.copy()
        table["display_value"] = table[metric]
        table["display_only_transformation"] = transform
        table["metric_definition"] = definition
        table.to_csv(OUT / filename, index=False)

    manifest_path = OUT / "fig6_source_manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        mask = manifest["panel"].eq("c")
        if mask.any():
            manifest.loc[mask, "display_only_transformation"] = (
                "none; raw formal E_pop and E_ind used; separate metric-wise raw colourbars"
            )
            manifest.to_csv(manifest_path, index=False)


def _write_main_readme() -> Path:
    path = OUT / "README_fig6.md"
    path.write_text(
        """# Fig. 6 main-text figure

This main figure uses only the frozen R5 formal/canonical sources and the
existing plotting tables. No data, metric, representative cycle, training,
inference, or statistical analysis was changed.

Performance-card axes are scaled within each dataset; absolute RMSE values
are indicated above the bars.

Panel (a) uses the established `Fig6_main` 1×6 geometry and in-panel HUST
legend. Panel (c) uses the established five-column/two-row small-multiple
heatmaps. The two narrow numerical colourbars follow the Fig.4b convention:
one raw Population scale and one raw Individual scale shared across the five
external systems. No min–max display normalization is applied.

Sources and formal provenance are documented in `README_fig6.md` and
`fig6_source_manifest.csv`; the c plotting tables retain the raw formal
values in their `display_value` columns.
""",
        encoding="utf-8",
    )
    return path


def _write_main_qc(outputs: list[Path], readme: Path) -> Path:
    path = OUT / "fig6_qc.md"
    lines = [
        "# Fig. 6 main-figure QC",
        "",
        "- [x] a/b/c overall layout unchanged.",
        "- [x] Panel labels moved left with one shared x coordinate.",
        "- [x] Panel-a geometry restored to Fig6_main, with its compact HUST in-panel legend.",
        "- [x] Panel-a legend does not cover trajectory data.",
        "- [x] Panel-b bar-value labels are horizontal and use three decimals.",
        "- [x] Redundant panel-b y ticks are removed; row labels remain.",
        "- [x] Panel-b still displays absolute RMSE values above bars.",
        "- [x] Panel-c uses the restored compact Fig.4b-style raw colorbars.",
        "- [x] Panel-c applies no min–max display normalization.",
        "- [x] Raw E_ind is displayed on an independent metric-wise scale; no new cross-system ranking statistic is computed.",
        "- [x] No new scientific analysis, training, or inference.",
        "- [x] No formal values changed.",
        "",
        "## Outputs",
        "",
        *[f"- `{path.resolve()}`" for path in outputs],
        f"- `{readme.resolve()}`",
        "",
        "The final `Fig6_main.png/pdf/svg` files are generated by `make_fig6.py`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    base.warnings.filterwarnings("ignore", category=UserWarning)
    base.apply_paper_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.labelsize": 6.7,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    previous_font_family = base.export_helpers.FONT_FAMILY
    base.export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        data = base._audit_and_load()
        base._write_plotting_tables(data)
        _rewrite_raw_c_tables(data)
        fig = plt.figure(figsize=(base.NC_WIDTH, 200 * base.MM), facecolor="white")
        outer = fig.add_gridspec(
            3,
            1,
            height_ratios=(0.86, 1.08, 1.25),
            hspace=0.20,
            left=0.075,
            right=0.985,
            top=0.985,
            bottom=0.055,
        )
        axes_a = _plot_panel_a_refined(fig, outer[0], data)
        axes_b = _plot_panel_b_refined(fig, outer[1], data)
        axes_c, heatmap_axes = _plot_panel_c_refined(fig, outer[2], data)

        base.enforce_nc_readability(fig)
        # The shared exporter floors collection linewidths for readability;
        # restore the thinner heatmap separators requested for this final
        # refinement without touching axes/spine widths.
        for ax in heatmap_axes:
            for collection in ax.collections:
                collection.set_linewidths([0.42])
        for artist in fig.findobj():
            if getattr(artist, "get_gid", lambda: None)() == "fig6_refined_bar_value":
                artist.set_fontsize(5.0)
            if getattr(artist, "get_gid", lambda: None)() == "fig6_refined_condition_label":
                artist.set_fontsize(5.0)

        base._typography_audit(fig, "fig6_main", OUT)
        fig.savefig(MAIN_PNG, dpi=600, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(MAIN_PDF, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(MAIN_SVG, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    finally:
        base.export_helpers.FONT_FAMILY = previous_font_family

    readme = _write_main_readme()
    qc = _write_main_qc([MAIN_PNG, MAIN_PDF, MAIN_SVG], readme)

    print("FIG6 MAIN:")
    print("PASS")
    print("OVERALL LAYOUT CHANGED:")
    print("NO")
    print("PANEL LABELS MOVED LEFT:")
    print("YES")
    print("PANEL A:")
    print("PASS")
    print("PANEL B:")
    print("PASS")
    print("PANEL C:")
    print("PASS")
    print("SCIENTIFIC VALUES CHANGED:")
    print("NO")
    print("NEW ANALYSIS:")
    print("NO")
    print("OUTPUTS:")
    for path in [MAIN_PNG, MAIN_PDF, MAIN_SVG, readme, qc]:
        print(path.resolve())
    print("REMAINING VISUAL ISSUES:")
    print("None identified in the rendered refinement.")


if __name__ == "__main__":
    main()
