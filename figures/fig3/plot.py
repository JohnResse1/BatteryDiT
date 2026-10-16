"""Draw progressive observation-degradation stress and signature readouts.

Pure figure layer: every number comes from `outputs/analysis/`, which is
produced by `python main.py --analysis stress_test`.  This module never reads
`results/` and never computes a statistic.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

from figures.common.tables import read_arrays, read_json, read_table

from figures.common import style
from figures.common.style import COLORS, FIGURE_WIDTH, apply_paper_style, panel_label
from figures.common.export import export_figure


FIGURE_ID = "fig3"


LEVELS = np.arange(1, 6)
LEVEL_LABELS = tuple(f"L{level}" for level in LEVELS)
CONDITIONS = OrderedDict([
    ("downsample", "Downsampling"),
    ("random_missing", "Random-point missing"),
    ("segment_missing", "Segment missing"),
    ("destructive", "Destructive"),
    ("hybrid", "Hybrid"),
])
MODELS = OrderedDict([
    ("batterydit", "BatteryDiT"),
    ("csdi", "CSDI"),
    ("transformer", "Transformer"),
    ("kalman", "Kalman"),
])
MODEL_STYLES = OrderedDict([
    ("batterydit", {"marker": "o", "markersize": 3.2, "linewidth": 1.2,
                    "linestyle": "-", "zorder": 4}),
    ("csdi", {"marker": "s", "markersize": 2.8, "linewidth": 1.05,
              "linestyle": "--", "zorder": 2}),
    ("transformer", {"marker": "^", "markersize": 2.8, "linewidth": 1.05,
                     "linestyle": ":", "zorder": 2}),
    ("kalman", {"marker": "D", "markersize": 2.6, "linewidth": 1.05,
                "linestyle": "-.", "zorder": 3}),
])
SOH_CMAP = LinearSegmentedColormap.from_list(
    "stress_soh", [style.C_CORRUPTED, "#EAA88C", "#ABBDC6", style.C_RECONSTRUCTED, style.C_CLEAN]
)


def _load():
    """Everything the figure needs, straight from the analysis layer."""
    curves = read_table("stress_test_curves.csv")
    relation = read_table("stress_test_signature_raw.csv")
    arrays = read_arrays("stress_test_figure_arrays.npz")
    return curves, relation, arrays, read_json("stress_test_statistics.json")


def _curves_from_arrays(arrays, task: str) -> dict:
    labels = [str(value) for value in arrays["models"]]
    mean = arrays[f"{task}_mean"]
    sd = arrays[f"{task}_sd"]
    return OrderedDict((label, (mean[i], sd[i], mean[i].reshape(1, *mean[i].shape)))
                       for i, label in enumerate(labels))


def _draw_metric_row(axes, curves, ylabel: str, panel: str, *, log_scale: bool = False) -> None:
    model_keys = tuple(MODELS)
    for condition_index, (_, title) in enumerate(CONDITIONS.items()):
        axis = axes[condition_index]
        for model_index, (label, (mean, deviation, _)) in enumerate(curves.items()):
            if not np.isfinite(mean).all():
                continue
            color = COLORS[model_keys[model_index].lower()]
            model_style = MODEL_STYLES[model_keys[model_index]]
            axis.plot(LEVELS, mean[condition_index], color=color, label=label,
                      marker=model_style["marker"],
                      markersize=model_style["markersize"],
                      linewidth=model_style["linewidth"],
                      linestyle=model_style["linestyle"],
                      zorder=model_style["zorder"])
            # Draw the seed-level band directly; a zero-sd series (e.g. the
            # deterministic CSDI destructive pass) correctly has no band.
            lower = mean[condition_index] - deviation[condition_index]
            if log_scale:
                lower = np.maximum(lower, 1e-8)
            axis.fill_between(LEVELS, lower, mean[condition_index] + deviation[condition_index],
                              color=color, alpha=0.38, linewidth=0, zorder=1)
        axis.set_title(title)
        axis.set_xticks(LEVELS)
        axis.set_xticklabels(LEVEL_LABELS)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.12, linewidth=0.5)
        if condition_index == 0:
            axis.set_ylabel(ylabel)
            panel_label(axis, panel, x=-0.30)
        else:
            axis.tick_params(axis="y", labelleft=False)


def draw(dataset: str = "mit", *, results_root=None) -> dict[str, str]:
    """Build all five panels of the progressive-stress figure."""
    del results_root  # figure layer reads only outputs/analysis/
    apply_paper_style()
    curves, relation, arrays, statistics = _load()
    reconstruction = _curves_from_arrays(arrays, "reconstruction")
    prediction = _curves_from_arrays(arrays, "prediction")

    figure = plt.figure(figsize=(FIGURE_WIDTH, 190 / 25.4))
    outer = gridspec.GridSpec(4, 1, figure=figure, height_ratios=(1.0, 1.0, 1.0, 1.08), hspace=0.45)
    axes_a_spec = outer[0].subgridspec(1, 5, wspace=0.10)
    axes_a = []
    for index in range(5):
        axes_a.append(figure.add_subplot(axes_a_spec[0, index],
                                         sharey=axes_a[0] if index else None))
    axes_b_spec = outer[1].subgridspec(1, 5, wspace=0.10)
    axes_b = []
    for index in range(5):
        axes_b.append(figure.add_subplot(axes_b_spec[0, index],
                                         sharey=axes_b[0] if index else None))
    _draw_metric_row(axes_a, reconstruction, "Waveform RMSE (V)", "a", log_scale=True)
    _draw_metric_row(axes_b, prediction, "SOH RMSE", "b")
    axes_a[0].set_ylim(0.0055, 22.0)
    axes_a[0].legend(ncol=2, loc="upper left", bbox_to_anchor=(0.025, 0.99),
                     fontsize=4.0, handlelength=1.15, columnspacing=0.55,
                     handletextpad=0.28, frameon=True, facecolor="white",
                     edgecolor="none", framealpha=0.80)
    for axis in axes_a:
        axis.set_xlabel("Severity")
    axes_b[0].legend(ncol=2, loc="upper left", fontsize=4.0,
                     handlelength=1.25, columnspacing=0.65, handletextpad=0.3)
    for axis in axes_b:
        axis.set_xlabel("Severity")

    # Degradation-order recovery is the mechanism row (c), immediately after
    # the stress rows.  PCA and tau/RMSE relation follow as (d) and (e).
    order_axes = [figure.add_subplot(outer[2].subgridspec(1, 5, wspace=0.10)[0, index]) for index in range(5)]
    clean_tau = arrays["clean_tau"]
    corrupted_tau = arrays["corrupted_tau"]
    reconstructed_tau_all = arrays["reconstructed_tau"]
    for ci, (_, title) in enumerate(CONDITIONS.items()):
        axis = order_axes[ci]
        corrupted_mean = corrupted_tau[:, ci].mean(axis=0)
        corrupted_std = corrupted_tau[:, ci].std(axis=0, ddof=1)
        recovered_mean = reconstructed_tau_all[:, ci].mean(axis=0)
        recovered_std = reconstructed_tau_all[:, ci].std(axis=0, ddof=1)
        axis.axhline(clean_tau.mean(), color="#202020", linewidth=0.85, linestyle=":", label="Clean")
        axis.fill_between(LEVELS, corrupted_mean - corrupted_std, corrupted_mean + corrupted_std,
                          color=style.C_CORRUPTED, alpha=0.12, linewidth=0)
        axis.plot(LEVELS, corrupted_mean, color=style.C_CORRUPTED, marker="x",
                  markersize=3.2, linewidth=1.0, label="Corrupted")
        axis.fill_between(LEVELS, recovered_mean - recovered_std, recovered_mean + recovered_std,
                          color="#5693C4", alpha=0.14, linewidth=0)
        axis.plot(LEVELS, recovered_mean, color="#5693C4", marker="o",
                  markersize=2.8, linewidth=1.1, label="Reconstructed")
        axis.set_title(title)
        axis.set_xticks(LEVELS); axis.set_xticklabels(LEVEL_LABELS)
        axis.set_xlim(0.75, 5.25)
        axis.set_ylim(-0.05, 1.35); axis.set_xlabel("Severity")
        axis.grid(axis="y", alpha=0.12, linewidth=0.5)
        if ci == 0:
            axis.set_ylabel("Degradation-order Kendall \u03c4")
            panel_label(axis, "c", x=-0.30)
            axis.legend(loc="upper left", bbox_to_anchor=(0.03, 0.99),
                        fontsize=4.0, handlelength=1.1, columnspacing=0.45,
                        handletextpad=0.25, frameon=True, facecolor="white",
                        edgecolor="none", framealpha=0.80)
        else:
            axis.tick_params(axis="y", labelleft=False)

    middle = outer[3].subgridspec(1, 9, width_ratios=(0.66, 0.08, 0.66, 0.08, 0.66, 0.05, 0.065, 0.48, 1.18), wspace=0.0)
    pca_axes = [figure.add_subplot(middle[0, index]) for index in (0, 2, 4)]
    colorbar_axis = figure.add_subplot(middle[0, 6])
    relation_axis = figure.add_subplot(middle[0, 8])
    pca_values = (arrays["representative_pca_clean"], arrays["representative_pca_corrupted"],
                  arrays["representative_pca_reconstructed"])
    pca_titles = ("Clean", "Hybrid L3", "Reconstructed")
    soh = arrays["representative_soh"]
    x_limits = (min(float(v[:, 0].min()) for v in pca_values), max(float(v[:, 0].max()) for v in pca_values))
    y_limits = (min(float(v[:, 1].min()) for v in pca_values), max(float(v[:, 1].max()) for v in pca_values))
    scatter = None
    for index, (axis, values, title) in enumerate(zip(pca_axes, pca_values, pca_titles)):
        scatter = axis.scatter(values[:, 0], values[:, 1], c=soh, cmap=SOH_CMAP,
                               vmin=float(soh.min()), vmax=float(soh.max()),
                               s=4.2, alpha=0.55, linewidth=0)
        axis.set_title(title); axis.set_xlim(*x_limits); axis.set_ylim(*y_limits)
        axis.set_xlabel("PC1")
        if index == 0:
            axis.set_ylabel("PC2"); panel_label(axis, "d", x=-0.35)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.grid(alpha=0.08, linewidth=0.3)
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=4.2, length=1.5, pad=1.2)
    colorbar.set_label("Norm SOH RMSE", fontsize=4.5, labelpad=4.5,
                       rotation=90, va="center", ha="center")
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")

    # relation panel: numbers straight from the analysis table
    reconstructed_tau = arrays["reconstructed_tau"]
    x_mean = reconstructed_tau.mean(axis=0)
    y_mean = prediction["BatteryDiT"][0]
    relationship = statistics["signature_statistics"]["relationship"]
    rho = relationship["spearman_rho"]
    p_value = relationship["p_value"]
    condition_colors = (style.C_CLEAN, style.C_RECONSTRUCTED, "#ABBDC6", "#EAA88C", style.C_CORRUPTED)
    severity_markers = ("o", "s", "^", "D", "P")
    for ci, color in enumerate(condition_colors):
        for level in range(5):
            relation_axis.scatter(x_mean[ci, level], y_mean[ci, level], color=color,
                                  marker=severity_markers[level], s=22,
                                  edgecolor="white", linewidth=0.45)
    line_x = np.linspace(x_mean.min(), x_mean.max(), 100)
    slope = relationship["slope"]; intercept = relationship["intercept"]
    relation_axis.plot(line_x, intercept + slope * line_x, color="#555555",
                       linewidth=0.8, linestyle="--")
    relation_axis.text(0.96, 0.96, f"Spearman \u03c1 = {rho:.2f}\nP = {p_value:.2g}",
                       transform=relation_axis.transAxes, ha="right", va="top", fontsize=5)
    relation_axis.set_xlabel("Reconstructed degradation-order Kendall \u03c4")
    relation_axis.set_ylabel("SOH RMSE")
    relation_axis.set_title("Recovery structure vs SOH error")
    relation_axis.grid(alpha=0.08, linewidth=0.3)
    panel_label(relation_axis, "e", x=-0.12)
    severity_legend = relation_axis.legend(handles=[
        Line2D([], [], marker=marker, linestyle="", color="#777777",
               label=f"L{level + 1}", markersize=2.8)
        for level, marker in enumerate(("o", "s", "^", "D", "P"))
    ], loc="lower left", bbox_to_anchor=(0.015, 0.30), ncol=2,
       fontsize=3.6, columnspacing=0.26, handletextpad=0.16,
       labelspacing=0.20, borderaxespad=0.12, frameon=True,
       facecolor="white", edgecolor="none", framealpha=0.78)
    relation_axis.add_artist(severity_legend)
    condition_legend = relation_axis.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=3.5)
        for (_, label), color in zip(CONDITIONS.items(), condition_colors)
    ], loc="lower left", bbox_to_anchor=(0.015, 0.03), ncol=2, fontsize=3.8,
       columnspacing=0.4, handletextpad=0.24, labelspacing=0.20,
       borderaxespad=0.12, frameon=True, facecolor="white",
       edgecolor="none", framealpha=0.78)

    figure.subplots_adjust(left=0.075, right=0.995, top=0.975, bottom=0.065)
    outputs = export_figure(figure, FIGURE_ID)
    plt.close(figure)
    return outputs
