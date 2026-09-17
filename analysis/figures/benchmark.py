"""Draw the formal heterogeneous-corruption benchmark from completed runs."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np

from common.paths import RESULTS_ROOT
from common.plotting import COLORS, FIGURE_WIDTH, apply_paper_style, export_figure, panel_label
from common.results import FORMAL_CONDITIONS, condition_matrix, mean_and_std, require_runs, select_runs


CONDITION_LABELS = OrderedDict([
    ("clean", "Clean"), ("downsample", "Downsample"),
    ("random", "Random"), ("segment", "Segment"),
    ("destructive", "Destructive"),
])
DEGRADED_CONDITIONS = ("downsample", "random", "segment", "destructive")
RECONSTRUCTION_MODELS = OrderedDict([
    ("batterydit", "BatteryDiT"), ("csdi", "CSDI"),
    ("transformer", "Transformer"), ("kalman", "Kalman"),
])
DIRECT_MODELS = OrderedDict([
    ("gru", "GRU"), ("lstm", "LSTM"), ("transformer", "Transformer"),
    ("tcn", "TCN"), ("trb", "TRB"),
])


def _formal_runs(dataset: str, results_root: Path):
    reconstruction, assisted, direct = OrderedDict(), OrderedDict(), OrderedDict()
    for model, label in RECONSTRUCTION_MODELS.items():
        reconstruction[model] = require_runs(
            select_runs("Reconstruction", dataset, recon_model=model, results_root=results_root),
            f"{label} reconstruction", 1 if model == "kalman" else 3,
        )
        assisted[model] = require_runs(
            select_runs("PredictDownstream", dataset, recon_model=model, predictor="trb", results_root=results_root),
            f"{label}→TRB", 3,
        )
    for model, label in DIRECT_MODELS.items():
        direct[model] = require_runs(
            select_runs("DirectPrediction", dataset, direct_model=model, results_root=results_root),
            f"direct {label}", 3,
        )
    return reconstruction, assisted, direct


def _summaries(runs, metric: str):
    return OrderedDict((model, mean_and_std(condition_matrix(items, metric))) for model, items in runs.items())


def _seed_run(runs, seed: int = 2021):
    matching = [run for run in runs if run.seed == seed]
    if len(matching) != 1:
        raise FileNotFoundError(f"expected one seed {seed} run, found {[run.seed for run in runs]}")
    return matching[0]


def _trajectory(run, condition: str) -> dict[str, np.ndarray]:
    path = run.directory / f"reconstructions_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing full trajectory output: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"input_voltage", "clean_voltage", "reconstruction", "keys"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def _representatives(reconstruction_runs, reconstruction_summary):
    selected = {}
    batterydit = _seed_run(reconstruction_runs["batterydit"])
    for condition in DEGRADED_CONDITIONS:
        ours = _trajectory(batterydit, condition)
        errors = np.sqrt(np.mean((ours["reconstruction"] - ours["clean_voltage"]) ** 2, axis=(1, 2)))
        index = int(np.argmin(np.abs(errors - np.median(errors))))
        key = str(ours["keys"][index])
        condition_index = FORMAL_CONDITIONS.index(condition)
        baseline_model = min(
            (model for model in RECONSTRUCTION_MODELS if model != "batterydit"),
            key=lambda model: reconstruction_summary[model][0][condition_index],
        )
        baseline = _trajectory(_seed_run(reconstruction_runs[baseline_model]), condition)
        baseline_lookup = {str(value): position for position, value in enumerate(baseline["keys"])}
        if key not in baseline_lookup:
            raise ValueError(f"representative key {key} is absent from {baseline_model}/{condition}")
        selected[condition] = {
            "clean": ours["clean_voltage"][index].reshape(-1),
            "observed": ours["input_voltage"][index].reshape(-1),
            "batterydit": ours["reconstruction"][index].reshape(-1),
            "baseline": baseline["reconstruction"][baseline_lookup[key]].reshape(-1),
        }
    return selected


def _batterydit_cycle_errors(assisted_runs, reconstruction_runs):
    prediction_run = _seed_run(assisted_runs["batterydit"])
    reconstruction_run = _seed_run(reconstruction_runs["batterydit"])
    path = prediction_run.directory / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing prediction arrays: {path}")
    output = OrderedDict()
    with np.load(path, allow_pickle=False) as archive:
        for condition in CONDITION_LABELS:
            prediction = np.asarray(archive[f"{condition}_prediction"])[:, 0]
            target = np.asarray(archive[f"{condition}_target"])[:, 0]
            key_name = f"{condition}_keys"
            keys = (
                np.asarray(archive[key_name]).astype(str)
                if key_name in archive.files
                else _trajectory(reconstruction_run, condition)["keys"].astype(str)
            )
            if len(keys) != len(prediction):
                raise ValueError(f"prediction/key count mismatch for {condition}")
            output[condition] = (np.abs(prediction - target), keys)
    return output


def _whole_battery_ci(errors: np.ndarray, keys: np.ndarray, seed: int) -> tuple[float, float, float]:
    cells = np.asarray([key.split("/", 1)[0] for key in keys])
    cell_values = np.asarray([errors[cells == cell].mean() for cell in sorted(set(cells))])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cell_values), size=(10_000, len(cell_values)))
    bootstrap = cell_values[draws].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return float(errors.mean()), float(low), float(high)


def _grouped_bars(axis, summaries, model_order, conditions, ylabel: str, *, log_scale: bool = False):
    x = np.arange(len(conditions))
    width = 0.76 / len(model_order)
    columns = [FORMAL_CONDITIONS.index(condition) for condition in conditions]
    for index, model in enumerate(model_order):
        mean, deviation = summaries[model]
        offset = (index - (len(model_order) - 1) / 2) * width
        axis.bar(
            x + offset, mean[columns], width, yerr=deviation[columns], capsize=1.3,
            color=COLORS[model], alpha=0.95 if model == "batterydit" else 0.68,
            label=RECONSTRUCTION_MODELS.get(model, DIRECT_MODELS.get(model, model)), linewidth=0,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([CONDITION_LABELS[name] for name in conditions], rotation=28, ha="right")
    axis.set_ylabel(ylabel)
    if log_scale:
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.12, linewidth=0.5)
    axis.set_axisbelow(True)


def draw(dataset: str = "mit", *, results_root: Path = RESULTS_ROOT) -> dict[str, str]:
    """Build all five benchmark panels from formal result files."""
    apply_paper_style()
    reconstruction_runs, assisted_runs, direct_runs = _formal_runs(dataset, Path(results_root))
    reconstruction = _summaries(reconstruction_runs, "battery_balanced_rmse")
    assisted = _summaries(assisted_runs, "soh_rmse")
    direct = _summaries(direct_runs, "soh_rmse")
    representatives = _representatives(reconstruction_runs, reconstruction)
    cycle_errors = _batterydit_cycle_errors(assisted_runs, reconstruction_runs)

    figure = plt.figure(figsize=(FIGURE_WIDTH, 205 / 25.4))
    outer = gridspec.GridSpec(3, 1, figure=figure, height_ratios=(0.32, 0.34, 0.34), hspace=0.48)

    top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.16)
    top_axes = []
    for column, condition in enumerate(DEGRADED_CONDITIONS):
        axis = figure.add_subplot(top[column], sharey=top_axes[0] if top_axes else None)
        top_axes.append(axis)
        item = representatives[condition]
        x = np.arange(len(item["clean"]))
        valid = item["observed"] > 1e-6
        axis.plot(x[valid], item["observed"][valid], color="#888888", linewidth=0.65, alpha=0.65)
        axis.plot(x, item["baseline"], color="#C99B8A", linewidth=0.9)
        axis.plot(x, item["batterydit"], color=COLORS["batterydit"], linewidth=1.15)
        axis.plot(x, item["clean"], color="#202020", linewidth=0.8)
        axis.set_title(CONDITION_LABELS[condition])
        axis.set_xlabel("Charging step")
        axis.set_xlim(0, len(x) - 1)
        axis.grid(alpha=0.08, linewidth=0.4)
        if column == 0:
            axis.set_ylabel("Voltage (V)")
            panel_label(axis, "a")
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
    _grouped_bars(axis_b, reconstruction, tuple(RECONSTRUCTION_MODELS), DEGRADED_CONDITIONS, "Waveform RMSE (V)", log_scale=True)
    axis_b.legend(ncol=2, loc="upper left")
    panel_label(axis_b, "b")

    axis_c = figure.add_subplot(middle[1])
    _grouped_bars(axis_c, assisted, tuple(RECONSTRUCTION_MODELS), tuple(CONDITION_LABELS), "SOH RMSE")
    axis_c.legend(ncol=2, loc="upper left")
    panel_label(axis_c, "c")

    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=(0.52, 0.48), wspace=0.34)
    joint = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=bottom[0], width_ratios=(0.30, 0.70), wspace=0.08)
    axis_direct = figure.add_subplot(joint[0])
    degraded_columns = [FORMAL_CONDITIONS.index(name) for name in DEGRADED_CONDITIONS]
    for model in DIRECT_MODELS:
        value = float(direct[model][0][degraded_columns].mean())
        axis_direct.scatter(0, value, color=COLORS[model], s=25)
        axis_direct.annotate(DIRECT_MODELS[model], (0, value), xytext=(4, 0), textcoords="offset points", va="center", fontsize=5.2)
    axis_direct.set_xlim(-0.5, 1.4)
    axis_direct.set_xticks([])
    axis_direct.set_xlabel("No reconstruction")
    axis_direct.set_ylabel("Mean SOH RMSE")
    axis_direct.set_title("Direct prediction")
    axis_direct.grid(axis="y", alpha=0.12)
    panel_label(axis_direct, "d")

    axis_joint = figure.add_subplot(joint[1], sharey=axis_direct)
    for model, label in RECONSTRUCTION_MODELS.items():
        x_value = float(reconstruction[model][0][degraded_columns].mean())
        y_value = float(assisted[model][0][degraded_columns].mean())
        axis_joint.scatter(x_value, y_value, color=COLORS[model], marker="*" if model == "batterydit" else "o", s=55 if model == "batterydit" else 32)
        axis_joint.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=5.2)
    axis_joint.set_xscale("log")
    axis_joint.set_xlabel("Mean waveform RMSE (V)")
    axis_joint.set_title("Reconstruction-assisted")
    axis_joint.tick_params(axis="y", labelleft=False)
    axis_joint.grid(alpha=0.12)

    axis_e = figure.add_subplot(bottom[1])
    values = [cycle_errors[condition][0] for condition in CONDITION_LABELS]
    boxes = axis_e.boxplot(values, positions=np.arange(1, 6), widths=0.55, showfliers=False, patch_artist=True)
    for box, color in zip(boxes["boxes"], ("#2166AC", "#4F86B5", "#7FA6C2", "#A9BDC9", "#C7AAA0")):
        box.set_facecolor(color)
        box.set_alpha(0.62)
        box.set_edgecolor("none")
    for index, condition in enumerate(CONDITION_LABELS, start=1):
        errors, keys = cycle_errors[condition]
        mean, low, high = _whole_battery_ci(errors, keys, 20260819 + index)
        axis_e.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="D", markersize=3.2, color="#111111", markerfacecolor="white", capsize=2)
    axis_e.set_xticks(np.arange(1, 6))
    axis_e.set_xticklabels(CONDITION_LABELS.values(), rotation=28, ha="right")
    axis_e.set_ylabel("Cycle SOH absolute error")
    axis_e.grid(axis="y", alpha=0.12)
    panel_label(axis_e, "e")

    figure.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.065)
    outputs = export_figure(figure, "benchmark")
    plt.close(figure)
    return outputs
