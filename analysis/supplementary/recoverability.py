"""Supplementary validation and fidelity figures for the multilevel analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.paths import AUDIT_ROOT, RESULTS_ROOT, create_output_directories
from common.plotting import FIGURE_WIDTH, apply_paper_style, export_figure
from figures.population_individual import BOOTSTRAPS, CONDITIONS, LEVELS, SYSTEMS, _load


def _inputs(results_root: Path):
    battery, summary, directories = _load(Path(results_root))
    return battery, summary, directories


def draw_manifold_validation(dataset: str = "all", *, results_root: Path = RESULTS_ROOT) -> dict[str, str]:
    del dataset
    apply_paper_style()
    _, _, directories = _inputs(Path(results_root))
    rows = []
    for dataset_name, (label, _) in SYSTEMS.items():
        table = pd.read_csv(directories[dataset_name] / "population_manifold_validation.csv")
        batteries = table[table.row_type == "selected_battery"].copy()
        if batteries.empty:
            raise ValueError(f"no per-battery manifold validation rows in {directories[dataset_name]}")
        rows.append({
            "dataset": label,
            "constant_rmse": float(batteries.constant_rmse.mean()),
            "manifold_rmse": float(batteries.selected_rmse.mean()),
            "improved_batteries": int((batteries.selected_rmse < batteries.constant_rmse).sum()),
            "validation_batteries": int(len(batteries)),
        })
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(128 / 25.4, 72 / 25.4))
    y = np.arange(len(frame))[::-1]
    for position, row in zip(y, frame.itertuples()):
        axis.plot([row.constant_rmse, row.manifold_rmse], [position, position], color="#C7C7C7", linewidth=0.9)
    axis.scatter(frame.constant_rmse, y, s=29, color="#A6A6A6", edgecolor="white", label="Constant baseline")
    axis.scatter(frame.manifold_rmse, y, s=34, color="#2166AC", edgecolor="white", label="SOH-conditioned manifold")
    axis.set_yticks(y)
    axis.set_yticklabels(frame.dataset)
    axis.set_xlabel("Validation waveform RMSE (V)")
    axis.grid(axis="x", alpha=0.15, linewidth=0.5)
    axis.legend(loc="lower right")
    for position, row in zip(y, frame.itertuples()):
        axis.annotate(f"{row.improved_batteries}/{row.validation_batteries}", (max(row.constant_rmse, row.manifold_rmse), position), xytext=(5, 0), textcoords="offset points", va="center", fontsize=5.2)
    figure.subplots_adjust(left=0.20, right=0.92, bottom=0.18, top=0.93)
    outputs = export_figure(figure, "supplementary_manifold_validation", supplementary=True)
    plt.close(figure)
    create_output_directories(); frame.to_csv(AUDIT_ROOT / "supplementary_manifold_validation.csv", index=False)
    return outputs


def draw_individual_fidelity(dataset: str = "all", *, results_root: Path = RESULTS_ROOT) -> dict[str, str]:
    del dataset
    apply_paper_style()
    _, summary, _ = _inputs(Path(results_root))
    labels = [value[0] for value in SYSTEMS.values()]
    cmap_c = LinearSegmentedColormap.from_list("direction", ["#F7FBFF", "#6BAED6", "#08306B"])
    cmap_q = LinearSegmentedColormap.from_list("amplitude", ["#2166AC", "#F7F7F7", "#B2182B"])
    q_min, q_max = float(summary.median_Q.min()), float(summary.median_Q.max())
    if not q_min < 1 < q_max:
        raise ValueError(f"amplitude-ratio range [{q_min}, {q_max}] does not straddle one")
    figure = plt.figure(figsize=(FIGURE_WIDTH, 79 / 25.4))
    grid = figure.add_gridspec(2, 6, width_ratios=(1, 1, 1, 1, 1, 0.065), left=0.07, right=0.98, bottom=0.13, top=0.94, wspace=0.075, hspace=0.36)
    image_c = image_q = None
    for column, (condition, title) in enumerate(CONDITIONS.items()):
        selected = summary[summary.condition == condition]
        c_values = selected.pivot(index="dataset", columns="severity", values="median_C_ind").reindex(index=labels, columns=LEVELS).to_numpy(float)
        q_values = selected.pivot(index="dataset", columns="severity", values="median_Q").reindex(index=labels, columns=LEVELS).to_numpy(float)
        axes = (figure.add_subplot(grid[0, column]), figure.add_subplot(grid[1, column]))
        image_c = axes[0].imshow(c_values, aspect="auto", cmap=cmap_c, vmin=0, vmax=1)
        image_q = axes[1].imshow(q_values, aspect="auto", cmap=cmap_q, norm=TwoSlopeNorm(vmin=q_min, vcenter=1, vmax=q_max))
        axes[0].set_title(title)
        for row, axis in enumerate(axes):
            axis.set_xticks(np.arange(5)); axis.set_xticklabels(LEVELS if row else [])
            axis.set_yticks(np.arange(6)); axis.set_yticklabels(labels if column == 0 else [])
            axis.tick_params(length=0)
        axes[1].set_xlabel("Severity")
    figure.colorbar(image_c, cax=figure.add_subplot(grid[0, 5])).set_label(r"Directional similarity, $C_{ind}$")
    figure.colorbar(image_q, cax=figure.add_subplot(grid[1, 5])).set_label(r"Amplitude ratio, $Q$")
    outputs = export_figure(figure, "supplementary_individual_fidelity", supplementary=True)
    plt.close(figure)
    return outputs


def _bootstrap(frame: pd.DataFrame, seed: int) -> tuple[float, float, float, int]:
    observed = float(spearmanr(frame.C_ind, frame.signature_error).statistic)
    groups = [group[["C_ind", "signature_error"]].to_numpy(float) for _, group in frame.groupby("cell", sort=True)]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAPS):
        selected = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[index] for index in selected])
        value = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if np.isfinite(value): estimates.append(value)
    low, high = np.percentile(estimates, (2.5, 97.5))
    return observed, float(low), float(high), len(estimates)


def draw_signature_linkage(dataset: str = "all", *, results_root: Path = RESULTS_ROOT) -> dict[str, str]:
    del dataset
    apply_paper_style()
    battery, _, _ = _inputs(Path(results_root))
    rows = []
    for index, label in enumerate(value[0] for value in SYSTEMS.values()):
        group = battery[battery.dataset == label]
        rho, low, high, finite = _bootstrap(group, 20260829 + 1000 * index)
        rows.append({"dataset": label, "spearman_rho": rho, "bootstrap_low": low, "bootstrap_high": high, "finite_bootstraps": finite, "n_batteries": int(group.cell.nunique())})
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(96 / 25.4, 73 / 25.4))
    y = np.arange(len(frame))[::-1]
    axis.errorbar(frame.spearman_rho, y, xerr=np.vstack((frame.spearman_rho - frame.bootstrap_low, frame.bootstrap_high - frame.spearman_rho)), fmt="o", markersize=5.2, color="#2878B5", markeredgecolor="white", capsize=2.5)
    axis.axvline(0, color="#777777", linestyle="--", linewidth=0.7)
    axis.set_yticks(y); axis.set_yticklabels(frame.dataset)
    axis.set_xlabel(r"Spearman $\rho(C_{ind},\,E_{sig})$")
    axis.grid(axis="x", alpha=0.16, linewidth=0.5)
    figure.subplots_adjust(left=0.21, right=0.95, bottom=0.18, top=0.94)
    outputs = export_figure(figure, "supplementary_signature_linkage", supplementary=True)
    plt.close(figure)
    create_output_directories(); frame.to_csv(AUDIT_ROOT / "supplementary_signature_linkage.csv", index=False)
    return outputs
