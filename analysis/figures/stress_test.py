"""Draw progressive observation-degradation stress and signature readouts."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import linregress, spearmanr

from common.paths import AUDIT_ROOT, RESULTS_ROOT, create_output_directories
from common.plotting import COLORS, FIGURE_WIDTH, apply_paper_style, export_figure, panel_label


LEVELS = np.arange(1, 6)
LEVEL_LABELS = tuple(f"L{level}" for level in LEVELS)
CONDITIONS = OrderedDict([
    ("downsample", "Downsample"),
    ("random_missing", "Random missing"),
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
MARKERS = ("o", "s", "^", "D")
SOH_CMAP = LinearSegmentedColormap.from_list(
    "stress_soh", ["#D6604D", "#EAA88C", "#ABBDC6", "#67A1CB", "#2166AC"]
)


def _payloads(group: str, dataset: str, filename: str, model: str, results_root: Path) -> list[dict]:
    root = results_root / group / dataset
    payloads = []
    if root.is_dir():
        for path in sorted(root.glob(f"*/{filename}")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("reconstruction_model", "")) == model:
                payloads.append(payload)
    return sorted(payloads, key=lambda item: int(item["model_seed"]))


def _require(payloads: list[dict], label: str, expected: int) -> list[dict]:
    if len(payloads) != expected:
        seeds = [item.get("model_seed") for item in payloads]
        raise FileNotFoundError(f"{label} requires {expected} stress result(s); found model seeds {seeds}")
    expected_seeds = [2021] if expected == 1 else [2021, 2022, 2023]
    seeds = [int(item["model_seed"]) for item in payloads]
    if seeds != expected_seeds:
        raise ValueError(f"{label} has model seeds {seeds}; expected {expected_seeds}")
    evaluation_seeds = {int(item["evaluation_seed"]) for item in payloads}
    if evaluation_seeds != {3407}:
        raise ValueError(f"{label} must use fixed evaluation seed 3407; found {sorted(evaluation_seeds)}")
    return payloads


def _curves(payloads: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([
        [[float(row[metric]) for row in payload[condition]] for condition in CONDITIONS]
        for payload in payloads
    ])
    if values.shape[1:] != (5, 5) or not np.isfinite(values).all():
        raise ValueError(f"invalid stress matrix for {metric}: {values.shape}")
    deviation = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])
    return values.mean(axis=0), deviation, values


def _load_stress(dataset: str, results_root: Path):
    reconstruction, prediction = OrderedDict(), OrderedDict()
    for model, label in MODELS.items():
        reconstruction[label] = _curves(
            _require(
                _payloads("ReconstructionStressTest", dataset, "stress_metrics.json", model, results_root),
                f"{label} reconstruction", 1 if model == "kalman" else 3,
            ),
            "sample_rmse_mean",
        )
        prediction[label] = _curves(
            _require(
                _payloads("PredictStressTest", dataset, "prediction_stress_metrics.json", model, results_root),
                f"{label}→TRB", 3,
            ),
            "soh_rmse",
        )
    return reconstruction, prediction


def _load_signatures(dataset: str, results_root: Path) -> list[dict[str, np.ndarray]]:
    paths = sorted((results_root / "DegradationSignature" / dataset).glob("*/degradation_signature.npz"))
    payloads = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]) for name in archive.files}
        if tuple(payload["topologies"].astype(str)) != tuple(CONDITIONS):
            raise ValueError(f"topology order mismatch in {path}")
        payloads.append(payload)
    payloads.sort(key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in payloads]
    if seeds != [2021, 2022, 2023]:
        raise FileNotFoundError(f"degradation signature requires model seeds 2021–2023; found {seeds}")
    if {int(item["evaluation_seed"]) for item in payloads} != {3407}:
        raise ValueError("degradation signatures must use fixed evaluation seed 3407")
    return payloads


def _draw_metric_row(axes, curves, ylabel: str, panel: str, *, log_scale: bool = False) -> None:
    model_keys = tuple(MODELS)
    for condition_index, (_, title) in enumerate(CONDITIONS.items()):
        axis = axes[condition_index]
        for model_index, (label, (mean, deviation, _)) in enumerate(curves.items()):
            color = COLORS[model_keys[model_index]]
            axis.plot(LEVELS, mean[condition_index], marker=MARKERS[model_index], color=color, label=label)
            if np.any(deviation[condition_index] > 0):
                lower = mean[condition_index] - deviation[condition_index]
                if log_scale:
                    lower = np.maximum(lower, 1e-8)
                axis.fill_between(LEVELS, lower, mean[condition_index] + deviation[condition_index], color=color, alpha=0.12, linewidth=0)
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


def draw(dataset: str = "mit", *, results_root: Path = RESULTS_ROOT) -> dict[str, str]:
    """Build all five panels of the progressive-stress figure."""
    apply_paper_style()
    results_root = Path(results_root)
    reconstruction, prediction = _load_stress(dataset, results_root)
    signatures = _load_signatures(dataset, results_root)

    figure = plt.figure(figsize=(FIGURE_WIDTH, 178 / 25.4))
    outer = gridspec.GridSpec(4, 1, figure=figure, height_ratios=(1.0, 1.0, 1.20, 1.0), hspace=0.48)
    axes_a = [figure.add_subplot(outer[0].subgridspec(1, 5, wspace=0.10)[0, index]) for index in range(5)]
    axes_b = [figure.add_subplot(outer[1].subgridspec(1, 5, wspace=0.10)[0, index]) for index in range(5)]
    _draw_metric_row(axes_a, reconstruction, "Waveform RMSE (V)", "a", log_scale=True)
    _draw_metric_row(axes_b, prediction, "SOH RMSE", "b")
    axes_a[0].legend(ncol=2, loc="upper left", fontsize=4.4)
    axes_b[0].legend(ncol=2, loc="upper left", fontsize=4.4)
    for axis in axes_b:
        axis.set_xlabel("Severity")

    middle = outer[2].subgridspec(1, 9, width_ratios=(1.0, 0.15, 1.0, 0.15, 1.0, 0.16, 0.05, 0.42, 2.05), wspace=0.0)
    pca_axes = [figure.add_subplot(middle[0, index]) for index in (0, 2, 4)]
    colorbar_axis = figure.add_subplot(middle[0, 6])
    relation_axis = figure.add_subplot(middle[0, 8])
    representative = signatures[0]
    pca_values = (
        representative["representative_pca_clean"],
        representative["representative_pca_corrupted"],
        representative["representative_pca_reconstructed"],
    )
    pca_titles = ("Clean", "Hybrid L3", "Reconstructed")
    soh = representative["representative_soh"]
    x_limits = (min(float(value[:, 0].min()) for value in pca_values), max(float(value[:, 0].max()) for value in pca_values))
    y_limits = (min(float(value[:, 1].min()) for value in pca_values), max(float(value[:, 1].max()) for value in pca_values))
    scatter = None
    for index, (axis, values, title) in enumerate(zip(pca_axes, pca_values, pca_titles)):
        scatter = axis.scatter(values[:, 0], values[:, 1], c=soh, cmap=SOH_CMAP, vmin=float(soh.min()), vmax=float(soh.max()), s=4.2, alpha=0.55, linewidth=0)
        axis.set_title(title)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("PC1")
        if index == 0:
            axis.set_ylabel("PC2")
            panel_label(axis, "c", x=-0.35)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.grid(alpha=0.08, linewidth=0.3)
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.ax.set_title("SOH", fontsize=5, pad=2)

    reconstructed_tau = np.stack([item["reconstructed_tau"] for item in signatures])
    x_mean = reconstructed_tau.mean(axis=0)
    y_mean = prediction["BatteryDiT"][0]
    x_values, y_values = x_mean.ravel(), y_mean.ravel()
    rho, p_value = spearmanr(x_values, y_values)
    condition_colors = ("#2166AC", "#67A1CB", "#ABBDC6", "#EAA88C", "#D6604D")
    severity_markers = ("o", "s", "^", "D", "P")
    for condition_index, ((_, label), color) in enumerate(zip(CONDITIONS.items(), condition_colors)):
        for level in range(5):
            relation_axis.scatter(x_mean[condition_index, level], y_mean[condition_index, level], color=color, marker=severity_markers[level], s=22, edgecolor="white", linewidth=0.45)
    regression = linregress(x_values, y_values)
    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    relation_axis.plot(line_x, regression.intercept + regression.slope * line_x, color="#555555", linewidth=0.8, linestyle="--")
    relation_axis.text(0.96, 0.96, f"Spearman ρ = {rho:.2f}\nP = {p_value:.2g}", transform=relation_axis.transAxes, ha="right", va="top", fontsize=5)
    relation_axis.set_xlabel("Reconstructed degradation-order Kendall τ")
    relation_axis.set_ylabel("SOH RMSE")
    relation_axis.set_title("Structure preservation vs diagnosis error")
    relation_axis.grid(alpha=0.08, linewidth=0.3)
    panel_label(relation_axis, "d", x=-0.12)
    relation_axis.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=3.5)
        for (_, label), color in zip(CONDITIONS.items(), condition_colors)
    ], loc="lower left", ncol=2, fontsize=3.9)

    order_axes = [figure.add_subplot(outer[3].subgridspec(1, 5, wspace=0.10)[0, index]) for index in range(5)]
    clean_tau = np.asarray([float(item["clean_tau"]) for item in signatures])
    corrupted_tau = np.stack([item["corrupted_tau"] for item in signatures])
    for condition_index, (_, title) in enumerate(CONDITIONS.items()):
        axis = order_axes[condition_index]
        corrupted_mean = corrupted_tau[:, condition_index].mean(axis=0)
        corrupted_std = corrupted_tau[:, condition_index].std(axis=0, ddof=1)
        recovered_mean = reconstructed_tau[:, condition_index].mean(axis=0)
        recovered_std = reconstructed_tau[:, condition_index].std(axis=0, ddof=1)
        axis.axhline(clean_tau.mean(), color="#202020", linewidth=0.85, linestyle=":", label="Clean")
        axis.fill_between(LEVELS, corrupted_mean - corrupted_std, corrupted_mean + corrupted_std, color="#D6604D", alpha=0.12, linewidth=0)
        axis.plot(LEVELS, corrupted_mean, color="#D6604D", marker="x", label="Corrupted")
        axis.fill_between(LEVELS, recovered_mean - recovered_std, recovered_mean + recovered_std, color="#5693C4", alpha=0.14, linewidth=0)
        axis.plot(LEVELS, recovered_mean, color="#5693C4", marker="o", label="Reconstructed")
        axis.set_title(title)
        axis.set_xticks(LEVELS)
        axis.set_xticklabels(LEVEL_LABELS)
        axis.set_ylim(-0.05, 1.0)
        axis.set_xlabel("Severity")
        axis.grid(axis="y", alpha=0.12, linewidth=0.5)
        if condition_index == 0:
            axis.set_ylabel("Degradation-order Kendall τ")
            panel_label(axis, "e", x=-0.30)
            axis.legend(loc="center left", fontsize=4.2)
        else:
            axis.tick_params(axis="y", labelleft=False)

    figure.subplots_adjust(left=0.075, right=0.995, top=0.975, bottom=0.065)
    outputs = export_figure(figure, "stress_test")
    plt.close(figure)
    create_output_directories()
    (AUDIT_ROOT / "stress_test_statistics.json").write_text(json.dumps({
        "dataset": dataset,
        "model_seeds": [2021, 2022, 2023],
        "evaluation_seed": 3407,
        "cycles_per_battery": int(representative["cycles_per_cell"]),
        "test_samples": int(len(representative["representative_soh"])),
        "relationship": {"spearman_rho": float(rho), "p_value": float(p_value), "n_condition_levels": int(len(x_values))},
    }, indent=2) + "\n", encoding="utf-8")
    return outputs
