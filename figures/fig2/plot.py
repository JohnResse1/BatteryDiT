"""Draw the formal heterogeneous-corruption benchmark.

Pure figure layer: every number comes from `outputs/analysis/`, produced by
`python main.py --analysis benchmark`.  This module never reads `results/` and
never computes a statistic.
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np

from figures.common.paths import figure_dir
from figures.common import style
from figures.common.style import COLORS, FIGURE_WIDTH, apply_paper_style, panel_label
from figures.common.tables import read_arrays, read_json, read_table
from figures.common.export import export_figure


FIGURE_ID = "fig2"
from figures.common import style
from figures.common.style import COLORS, FIGURE_WIDTH, apply_paper_style, panel_label
from figures.common.export import export_figure


FIGURE_ID = "fig2"


CONDITION_LABELS = OrderedDict([
    ("clean", "Clean"), ("downsample", "Downsampling"),
    ("random", "Random-point missing"), ("segment", "Segment missing"),
    ("destructive", "Destructive noise"),
])
TRAJECTORY_CONDITION_LABELS = {
    "downsample": "Downsampling",
    "random": "Random-point missing",
    "segment": "Segment missing",
    "destructive": "Destructive noise",
}
DEGRADED_CONDITIONS = ("downsample", "random", "segment", "destructive")
RECONSTRUCTION_MODELS = OrderedDict([
    ("batterydit", "BatteryDiT"), ("csdi", "CSDI"),
    ("transformer", "Transformer"), ("kalman", "Kalman"),
])
DIRECT_MODELS = OrderedDict([
    ("gru", "GRU"), ("lstm", "LSTM"), ("transformer", "Transformer"),
    ("tcn", "TCN"), ("trb", "TRB"),
])
FORMAL_CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")


def _load():
    """Summary/CI/error tables and representative curves from analysis."""
    summary = read_table("benchmark_summary.csv")
    ci = read_table("benchmark_soh_ci.csv")
    errors = read_table("benchmark_cycle_errors.csv")
    arrays = read_arrays("benchmark_representatives.npz")
    return summary, ci, errors, arrays


def _summaries(summary, task: str, models):
    """(mean, sd) per model over the formal condition order."""
    table = summary[summary.task == task]
    output = OrderedDict()
    for model in models:
        rows = table[table.model == model].set_index("condition").reindex(FORMAL_CONDITIONS)
        if rows.empty or rows.value_mean.isna().all():
            continue  # results pending (e.g. CSDI downstream prediction)
        output[model] = (rows.value_mean.to_numpy(float), rows.value_sd.to_numpy(float))
    return output


RECON_BAR_COLORS = {
    "batterydit": "#2166AC",
    "csdi": "#9FBBCF",
    "transformer": "#C6C3C1",
    "kalman": "#D4B2A6",
}
SOH_BAR_COLORS = {
    "batterydit": "#2166AC",
    "csdi": "#9EB7C9",
    "transformer": "#BCC9D1",
    "kalman": "#7899B3",
}


def _grouped_bars(axis, summaries, model_order, conditions, ylabel: str, *, log_scale: bool = False, palette=None):
    x = np.arange(len(conditions))
    columns = [FORMAL_CONDITIONS.index(condition) for condition in conditions]
    present_models = [model for model in model_order if model in summaries]
    width = (0.72 if log_scale else 0.76) / max(len(present_models), 1)
    for index, model in enumerate(present_models):
        mean, deviation = summaries[model]
        offset = (index - (len(present_models) - 1) / 2) * width
        color = (palette or {}).get(model, COLORS.get(model.lower(), COLORS.get(model, "#777777")))
        bars = axis.bar(
            x + offset, mean[columns], width, yerr=deviation[columns], capsize=1.3,
            color=color, edgecolor="white", linewidth=0.3,
            alpha=0.96 if model == "batterydit" else 0.62,
            label=RECONSTRUCTION_MODELS.get(model, DIRECT_MODELS.get(model, model)),
        )
        if model == "batterydit":
            for bar, value in zip(bars, mean[columns]):
                if log_scale:
                    axis.text(bar.get_x() + bar.get_width() / 2, value * 1.45,
                              f"{value:.3f}", ha="center", va="bottom", fontsize=4.3,
                              fontweight="bold", color=color)
                else:
                    axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                              f"{value:.3f}", ha="center", va="bottom", fontsize=4.2,
                              fontweight="bold", color=color)
    axis.set_xticks(x)
    axis.set_xticklabels([CONDITION_LABELS[name] for name in conditions], rotation=28, ha="right")
    axis.set_ylabel(ylabel)
    if log_scale:
        axis.set_yscale("log")
        axis.set_ylim(0.001, 3.0)
    axis.grid(axis="y", alpha=0.12, linewidth=0.5)
    axis.set_axisbelow(True)


def draw(dataset: str = "mit", *, results_root=None) -> dict[str, str]:
    """Build all five benchmark panels from the analysis layer."""
    del dataset, results_root
    apply_paper_style()
    summary, ci, errors, arrays = _load()
    reconstruction = _summaries(summary, "reconstruction", tuple(RECONSTRUCTION_MODELS))
    assisted = _summaries(summary, "assisted", tuple(RECONSTRUCTION_MODELS))
    direct = _summaries(summary, "direct", tuple(DIRECT_MODELS))

    figure = plt.figure(figsize=(FIGURE_WIDTH, 182 / 25.4))
    outer = gridspec.GridSpec(3, 1, figure=figure, height_ratios=(0.24, 0.38, 0.38), hspace=0.46)

    top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.16)
    top_axes = []
    for column, condition in enumerate(DEGRADED_CONDITIONS):
        axis = figure.add_subplot(top[column], sharey=top_axes[0] if top_axes else None)
        top_axes.append(axis)
        item = {name: arrays[f"{condition}_{name}"]
                for name in ("clean", "observed", "batterydit", "baseline")}
        x = np.arange(len(item["clean"]))
        valid = item["observed"] > 1e-6
        axis.plot(x[valid], item["observed"][valid], color="#888888", linewidth=0.65, alpha=0.65)
        axis.plot(x, item["baseline"], color="#C99B8A", linewidth=0.9)
        axis.plot(x, item["batterydit"], color=COLORS["batterydit"], linewidth=1.15)
        axis.plot(x, item["clean"], color="#202020", linewidth=0.8)
        axis.set_title(TRAJECTORY_CONDITION_LABELS[condition]); axis.set_xlabel("Charging step")
        axis.set_xlim(0, len(x) - 1); axis.set_ylim(2.0, 4.0); axis.grid(alpha=0.08, linewidth=0.4)
        if column == 0:
            axis.set_ylabel("Voltage (V)"); panel_label(axis, "a")
        else:
            axis.tick_params(axis="y", labelleft=False)
    top_axes[0].legend(handles=[
        Line2D([], [], color="#202020", label="Clean"),
        Line2D([], [], color="#888888", label="Observed"),
        Line2D([], [], color=COLORS["batterydit"], label="BatteryDiT"),
        Line2D([], [], color="#C99B8A", label="Best baseline"),
    ], loc="lower right", fontsize=5.2)

    middle = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=(0.44, 0.56), wspace=0.35)
    axis_b = figure.add_subplot(middle[0])
    _grouped_bars(axis_b, reconstruction, tuple(RECONSTRUCTION_MODELS), DEGRADED_CONDITIONS,
                  "Waveform RMSE (V)", log_scale=True, palette=RECON_BAR_COLORS)
    axis_b.legend(ncol=1, loc="upper left"); panel_label(axis_b, "b")
    axis_c = figure.add_subplot(middle[1])
    _grouped_bars(axis_c, assisted, tuple(RECONSTRUCTION_MODELS), tuple(CONDITION_LABELS),
                  "SOH RMSE", palette=SOH_BAR_COLORS)
    axis_c.set_ylim(0, 0.062)
    axis_c.legend(ncol=2, loc="upper left"); panel_label(axis_c, "c")

    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=(0.44, 0.56), wspace=0.34)
    joint = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=bottom[0], width_ratios=(0.30, 0.70), wspace=0.08)
    axis_direct = figure.add_subplot(joint[0])
    degraded_columns = [FORMAL_CONDITIONS.index(name) for name in DEGRADED_CONDITIONS]
    direct_label_offsets = {"gru": (4, 4), "lstm": (4, 0), "transformer": (4, -4),
                            "tcn": (4, 0), "trb": (4, 0)}
    for model in DIRECT_MODELS:
        value = float(direct[model][0][degraded_columns].mean())
        color = COLORS.get(model.lower(), COLORS.get(model, "#777777"))
        axis_direct.scatter(0, value, color=color, s=25)
        dx, dy = direct_label_offsets.get(model, (4, 0))
        axis_direct.annotate(DIRECT_MODELS[model], (0, value), xytext=(dx, dy),
                             textcoords="offset points", va="center", fontsize=5.2, color=color)
    axis_direct.set_xlim(-0.65, 1.55); axis_direct.set_xticks([])
    axis_direct.set_facecolor("#FAFAFA")
    axis_direct.spines["right"].set_color("#999999")
    axis_direct.spines["right"].set_linestyle("--")
    axis_direct.set_xlabel("No reconstruction"); axis_direct.set_ylabel("Mean SOH RMSE")
    axis_direct.set_title("Direct prediction", pad=3.0); axis_direct.grid(axis="y", alpha=0.12)
    panel_label(axis_direct, "d", x=-0.38, y=1.10)

    axis_joint = figure.add_subplot(joint[1], sharey=axis_direct)
    d_method_colors = {
        "batterydit": "#2166AC", "csdi": "#91AFC2",
        "transformer": "#BFB5B1", "kalman": "#C99F90",
    }
    label_offsets = {"batterydit": (8, 6), "csdi": (5, 6),
                     "transformer": (-7, 5), "kalman": (5, -9)}
    available_models = [m for m in RECONSTRUCTION_MODELS if m in reconstruction and m in assisted]
    for model in available_models:
        label = RECONSTRUCTION_MODELS[model]
        x_value = float(reconstruction[model][0][degraded_columns].mean())
        y_value = float(assisted[model][0][degraded_columns].mean())
        is_ours = model == "batterydit"
        axis_joint.scatter(x_value, y_value, color=d_method_colors[model],
                           marker="*" if is_ours else "o",
                           s=58 if is_ours else 38,
                           edgecolor="white", linewidth=0.55,
                           alpha=0.98 if is_ours else 0.70, zorder=3)
        dx, dy = label_offsets[model]
        axis_joint.annotate(label, (x_value, y_value), xytext=(dx, dy),
                            textcoords="offset points", fontsize=5,
                            fontweight="bold" if is_ours else "normal",
                            color=d_method_colors[model],
                            ha="left" if dx >= 0 else "right")
    axis_joint.set_xscale("log"); axis_joint.set_xlabel("Mean waveform RMSE (V)", labelpad=1)
    axis_joint.set_title("Reconstruction-assisted", pad=3.0)
    axis_joint.set_facecolor("white")
    axis_joint.tick_params(axis="y", left=False, labelleft=False); axis_joint.grid(alpha=0.12)

    # Panel e: waveform RMSE vs SOH RMSE for the four reconstruction-assisted
    # pipelines, one 2x2 panel per degraded corruption condition.
    grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=bottom[1], wspace=0.20, hspace=0.48)
    pipeline_labels = OrderedDict([
        ("batterydit", "BatteryDiT"), ("csdi", "CSDI"),
        ("kalman", "Kalman"), ("transformer", "Transformer"),
    ])
    e_axes = []
    for index, condition in enumerate(DEGRADED_CONDITIONS):
        row, column = divmod(index, 2)
        axis = figure.add_subplot(grid[row, column], sharex=e_axes[0] if e_axes else None,
                                  sharey=e_axes[0] if e_axes else None)
        e_axes.append(axis)
        # A light connecting path makes it easy to see whether better waveform
        # reconstruction tracks better SOH prediction within each corruption.
        ordered = ["transformer", "kalman", "csdi", "batterydit"]
        xs, ys = [], []
        for model, label in pipeline_labels.items():
            if model not in reconstruction or model not in assisted:
                continue
            column_index = FORMAL_CONDITIONS.index(condition)
            x_value = float(reconstruction[model][0][column_index])
            y_value = float(assisted[model][0][column_index])
            xs.append(x_value); ys.append(y_value)
            color = d_method_colors[model]
            axis.scatter(x_value, y_value, color=color, marker="o",
                         s=22, edgecolor="white", linewidth=0.45,
                         alpha=0.95 if model == "batterydit" else 0.80,
                         label=label if index == 0 else None,
                         zorder=3 if model == "batterydit" else 2)
        if len(xs) >= 2:
            order = np.argsort(xs)
            axis.plot(np.asarray(xs)[order], np.asarray(ys)[order],
                      color="#999999", linewidth=0.7, alpha=0.65,
                      linestyle="--", zorder=1)
        axis.set_xscale("log")
        axis.set_title(CONDITION_LABELS[condition], fontsize=6.0)
        axis.grid(True, alpha=0.10, linewidth=0.3)
        axis.tick_params(pad=1)
        if row == 1:
            axis.set_xlabel("Waveform RMSE (V)", fontsize=5.6, labelpad=1)
        if column == 0:
            axis.set_ylabel("SOH RMSE", fontsize=5.6, labelpad=1)
        else:
            axis.tick_params(axis="y", labelleft=False)
    e_axes[0].legend(loc="upper left", ncol=1, fontsize=4.2,
                     handlelength=0.9, handletextpad=0.28, borderaxespad=0.18,
                     labelspacing=0.24, frameon=False)
    e_axes[0].text(-0.30, 1.18, "(e)", transform=e_axes[0].transAxes,
                   fontsize=8, fontweight="bold", va="bottom", ha="left")

    figure.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.065)
    outputs = export_figure(figure, FIGURE_ID)
    plt.close(figure)
    return outputs
