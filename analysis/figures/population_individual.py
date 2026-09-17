"""Draw multilevel population and individual degradation recoverability."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.paths import AUDIT_ROOT, RESULTS_ROOT, create_output_directories
from common.plotting import FIGURE_WIDTH, apply_paper_style, export_figure


SYSTEMS = OrderedDict([
    ("mit", ("MIT", 2021)),
    ("hust", ("HUST", 2023)),
    ("mich", ("MICH", 2021)),
    ("stanford_2", ("Stanford", 2021)),
    ("tju_nca_45c", ("TJU-NCA", 2021)),
    ("tju_ncm_45c", ("TJU-NCM", 2021)),
])
CONDITIONS = OrderedDict([
    ("downsample", "Downsampling"),
    ("random_missing", "Random"),
    ("segment_missing", "Segment"),
    ("destructive", "Destructive"),
    ("hybrid", "Hybrid"),
])
LEVELS = tuple(f"L{level}" for level in range(1, 6))
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260829
Q_LOW, Q_HIGH = 0.95, 1.05
POP_CMAP = LinearSegmentedColormap.from_list("population_error", ["#F7FBFF", "#BED7E9", "#4E91C2", "#174A78"])
IND_CMAP = LinearSegmentedColormap.from_list("individual_error", ["#FFF8F0", "#F7CFAC", "#E78A56", "#A83032"])


def _run_directory(dataset: str, seed: int, results_root: Path) -> Path:
    root = results_root / "PopulationIndividual" / dataset
    matches = []
    if root.is_dir():
        for summary_path in sorted(root.glob("*/summary.json")):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(payload.get("model_seed", -1)) == seed and payload.get("status") == "COMPLETE":
                matches.append(summary_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"{dataset} population-individual seed {seed} requires one complete run; found {matches}")
    required = ("battery_level_decomposition.csv", "condition_summary.csv", "workflow_example.npz")
    missing = [name for name in required if not (matches[0] / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{matches[0]} is missing {missing}")
    return matches[0]


def _load(results_root: Path):
    battery_frames, summary_frames, directories = [], [], {}
    for dataset, (label, seed) in SYSTEMS.items():
        directory = _run_directory(dataset, seed, results_root)
        directories[dataset] = directory
        battery = pd.read_csv(directory / "battery_level_decomposition.csv")
        summary = pd.read_csv(directory / "condition_summary.csv")
        expected_conditions = set(CONDITIONS)
        if set(battery.condition) != expected_conditions or set(summary.condition) != expected_conditions:
            raise ValueError(f"{directory} has incomplete corruption topology coverage")
        if set(battery.severity) != set(LEVELS) or set(summary.severity) != set(LEVELS):
            raise ValueError(f"{directory} has incomplete severity coverage")
        coverage = battery.groupby("cell").size()
        if not coverage.eq(25).all() or len(summary) != 25:
            raise ValueError(f"{directory} does not contain 25 conditions per battery")
        battery.insert(0, "dataset", label)
        summary.insert(0, "dataset", label)
        battery_frames.append(battery)
        summary_frames.append(summary)
    return pd.concat(battery_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True), directories


def _cluster_bootstrap_rho(frame: pd.DataFrame, seed: int) -> tuple[float, float, float, int]:
    frame = frame[["cell", "E_ind", "soh_error"]].dropna()
    observed = float(spearmanr(frame.E_ind, frame.soh_error).statistic)
    batteries = sorted(frame.cell.astype(str).unique())
    groups = [frame[frame.cell.astype(str) == cell][["E_ind", "soh_error"]].to_numpy(float) for cell in batteries]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        selected = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[index] for index in selected])
        rho = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        raise RuntimeError("whole-battery Spearman bootstrap produced no finite value")
    low, high = np.percentile(values, (2.5, 97.5))
    return observed, float(low), float(high), len(values)


def _linkage(battery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(value[0] for value in SYSTEMS.values()):
        group = battery[battery.dataset == label]
        rho, low, high, finite = _cluster_bootstrap_rho(group, BOOTSTRAP_SEED + index)
        rows.append({
            "dataset": label, "spearman_rho": rho,
            "bootstrap_low": low, "bootstrap_high": high,
            "bootstrap_resamples": BOOTSTRAPS, "finite_bootstrap_resamples": finite,
            "n_batteries": int(group.cell.nunique()), "n_battery_conditions": int(len(group)),
        })
    return pd.DataFrame(rows)


def _failure_pathways(battery: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in (value[0] for value in SYSTEMS.values()):
        for condition in CONDITIONS:
            group = battery[(battery.dataset == dataset) & (battery.condition == condition)]
            medians = summary[(summary.dataset == dataset) & (summary.condition == condition)].sort_values("severity_index")
            wide = group.pivot(index="cell", columns="severity_index", values=["E_ind", "A_ind", "Q"])
            e_direction = wide[("E_ind", 5)] > wide[("E_ind", 1)]
            a_direction = wide[("A_ind", 5)] < wide[("A_ind", 1)]
            q_l5 = wide[("Q", 5)]
            q_path = medians.median_Q.to_numpy(float)
            a_path = medians.median_A_ind.to_numpy(float)
            e_path = medians.median_E_ind.to_numpy(float)
            modes = []
            if np.any(q_path < Q_LOW) and float((q_l5 < 1).mean()) > 0.5:
                modes.append("SHRINKAGE")
            if np.any(q_path > Q_HIGH) and e_path[-1] > e_path[0]:
                modes.append("AMPLIFICATION")
            if Q_LOW <= q_path[-1] <= Q_HIGH and a_path[-1] < a_path[0] and float(a_direction.mean()) > 0.5:
                modes.append("MISALIGNMENT")
            if len(modes) > 1:
                mode = "MIXED"
            elif modes:
                mode = modes[0]
            elif float(e_direction.mean()) <= 0.5 and float(a_direction.mean()) <= 0.5:
                mode = "PRESERVED"
            else:
                mode = "MIXED"
            rows.append({
                "dataset": dataset, "condition": condition, "failure_mode": mode,
                "E_ind_direction_fraction": float(e_direction.mean()),
                "A_ind_direction_fraction": float(a_direction.mean()),
            })
    return pd.DataFrame(rows)


def _medoids(summary: pd.DataFrame, pathways: pd.DataFrame) -> pd.DataFrame:
    frame = summary.merge(pathways[["dataset", "condition", "failure_mode"]], on=["dataset", "condition"], validate="many_to_one")
    q = frame.median_Q.to_numpy(float)
    a = frame.median_A_ind.to_numpy(float)
    q_iqr, a_iqr = np.subtract(*np.percentile(q, (75, 25))), np.subtract(*np.percentile(a, (75, 25)))
    if q_iqr <= 0 or a_iqr <= 0:
        raise RuntimeError("cannot IQR-standardize phenotype coordinates")
    frame["Q_iqr_z"] = (q - np.median(q)) / q_iqr
    frame["A_ind_iqr_z"] = (a - np.median(a)) / a_iqr
    frame["is_medoid"] = False
    for mode in ("SHRINKAGE", "MISALIGNMENT", "AMPLIFICATION"):
        indices = frame.index[(frame.failure_mode == mode) & (frame.severity == "L5")].to_numpy()
        if len(indices) == 0:
            raise RuntimeError(f"no L5 condition represents the formal {mode.lower()} phenotype")
        points = frame.loc[indices, ["Q_iqr_z", "A_ind_iqr_z"]].to_numpy(float)
        distance = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, axis=2)).sum(axis=1)
        frame.loc[int(indices[int(np.argmin(distance))]), "is_medoid"] = True
    return frame


def _workflow_panel(figure, spec, path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        item = {name: np.asarray(archive[name]) for name in archive.files}
    grid = spec.subgridspec(1, 4, width_ratios=(1.30, 1.15, 1.15, 0.72), wspace=0.34)
    axes = [figure.add_subplot(grid[index]) for index in range(4)]
    x = np.linspace(0, 1, len(item["clean"]))
    axes[0].plot(x, item["clean"], color="#111111", label="Clean")
    axes[0].plot(x, item["reconstructed"], color="#2166AC", label="Reconstructed")
    axes[0].set(title="Trajectory recovery", xlabel="Charging progress", ylabel="Voltage (V)")
    axes[0].legend(fontsize=4.5)
    axes[1].plot(x, item["clean"] - item["reference"], color="#111111", label=r"$d_{clean}$")
    axes[1].plot(x, item["reconstructed"] - item["reference"], color="#2166AC", label=r"$d_{rec}$")
    axes[1].axhline(0, color="#999999", linewidth=0.5)
    axes[1].set(title="Early-life reference", xlabel="Charging progress", ylabel="Deviation (V)")
    axes[1].legend(fontsize=4.5)
    axes[2].plot(x, item["clean_population"], color="#111111", label=r"$p(h)$")
    axes[2].plot(x, item["reconstructed_population"], color="#2166AC", label=r"$p(\hat h)$")
    axes[2].set(title="Population projection", xlabel="Charging progress", ylabel="Population component (V)")
    axes[2].legend(fontsize=4.5)
    q = np.linalg.norm(item["reconstructed_individual"]) / np.linalg.norm(item["clean_individual"])
    cosine = np.dot(item["reconstructed_individual"], item["clean_individual"]) / (np.linalg.norm(item["reconstructed_individual"]) * np.linalg.norm(item["clean_individual"]))
    axes[3].scatter([q], [cosine], s=35, color="#D6604D", edgecolor="white")
    axes[3].scatter([1], [1], marker="*", s=30, color="#222222")
    axes[3].axvline(1, color="#999999", linestyle="--", linewidth=0.6)
    axes[3].axhline(1, color="#999999", linestyle="--", linewidth=0.6)
    axes[3].set(title="Individual fidelity", xlabel="Amplitude ratio Q", ylabel="Directional cosine", xlim=(0, max(1.1, q * 1.15)), ylim=(-0.1, 1.05))
    for axis in axes:
        axis.grid(alpha=0.10, linewidth=0.4)


def _heatmaps(figure, spec, summary: pd.DataFrame) -> None:
    grid = spec.subgridspec(2, 6, width_ratios=(1, 1, 1, 1, 1, 0.065), hspace=0.36, wspace=0.075)
    labels = [value[0] for value in SYSTEMS.values()]
    pop_max, ind_max = float(summary.median_E_pop.max()), float(summary.median_E_ind.max())
    pop_image = ind_image = None
    for column, (condition, title) in enumerate(CONDITIONS.items()):
        selected = summary[summary.condition == condition]
        pop = selected.pivot(index="dataset", columns="severity", values="median_E_pop").reindex(index=labels, columns=LEVELS).to_numpy(float)
        ind = selected.pivot(index="dataset", columns="severity", values="median_E_ind").reindex(index=labels, columns=LEVELS).to_numpy(float)
        axes = (figure.add_subplot(grid[0, column]), figure.add_subplot(grid[1, column]))
        pop_image = axes[0].imshow(pop, aspect="auto", cmap=POP_CMAP, vmin=0, vmax=pop_max)
        ind_image = axes[1].imshow(ind, aspect="auto", cmap=IND_CMAP, vmin=0, vmax=ind_max)
        axes[0].set_title(title)
        for row, axis in enumerate(axes):
            axis.set_xticks(np.arange(5))
            axis.set_xticklabels(LEVELS if row == 1 else [])
            axis.set_yticks(np.arange(6))
            axis.set_yticklabels(labels if column == 0 else [])
            axis.tick_params(length=0)
        axes[1].set_xlabel("Severity")
    pop_bar, ind_bar = figure.colorbar(pop_image, cax=figure.add_subplot(grid[0, 5])), figure.colorbar(ind_image, cax=figure.add_subplot(grid[1, 5]))
    pop_bar.set_label(r"Population-stage error, $E_{pop}$")
    ind_bar.set_label(r"Individual-deviation error, $E_{ind}$")


def _linkage_panel(figure, spec, linkage: pd.DataFrame) -> None:
    axis = figure.add_subplot(spec)
    labels = [value[0] for value in SYSTEMS.values()]
    frame = linkage.set_index("dataset").reindex(labels)
    y = np.arange(len(labels))[::-1]
    rho, low, high = (frame[name].to_numpy(float) for name in ("spearman_rho", "bootstrap_low", "bootstrap_high"))
    axis.errorbar(rho, y, xerr=np.vstack((rho - low, high - rho)), fmt="o", markersize=5.2, color="#2878B5", markeredgecolor="white", capsize=2.5)
    axis.axvline(0, color="#777777", linestyle="--", linewidth=0.7)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlabel(r"Spearman $\rho(E_{ind},\,SOH\ error)$")
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)


def _phenotype_panel(figure, spec, phenotype: pd.DataFrame) -> None:
    axis = figure.add_subplot(spec)
    axis.scatter(phenotype.median_Q, phenotype.median_C_ind, s=8, color="#C9CDD2", alpha=0.38, edgecolors="none")
    axis.axvline(1, color="#777777", linestyle="--", linewidth=0.7)
    axis.axhline(1, color="#777777", linestyle="--", linewidth=0.7)
    styles = {
        "SHRINKAGE": ("Shrinkage", "#3B82B8", "o"),
        "MISALIGNMENT": ("Misalignment", "#E59A36", "s"),
        "AMPLIFICATION": ("Amplification", "#C94C4C", "D"),
    }
    for mode, (label, color, marker) in styles.items():
        row = phenotype[(phenotype.failure_mode == mode) & phenotype.is_medoid]
        item = row.iloc[0]
        axis.scatter(item.median_Q, item.median_C_ind, s=65, marker=marker, color=color, edgecolor="white", label=label)
        axis.annotate(label, (item.median_Q, item.median_C_ind), xytext=(4, 5), textcoords="offset points", color=color, fontsize=5.5)
    axis.scatter([1], [1], marker="*", s=28, color="#222222")
    axis.set_xlabel(r"Amplitude ratio $Q=\|e_{rec}\|/\|e_{clean}\|$")
    axis.set_ylabel(r"Directional cosine similarity $C_{ind}$")
    axis.grid(alpha=0.16, linewidth=0.5)
    axis.legend(loc="lower left")


def draw(dataset: str = "all", *, results_root: Path = RESULTS_ROOT) -> dict[str, str]:
    """Build Fig. 5 from six complete frozen multilevel analysis runs."""
    del dataset
    apply_paper_style()
    battery, summary, directories = _load(Path(results_root))
    linkage = _linkage(battery)
    pathways = _failure_pathways(battery, summary)
    phenotype = _medoids(summary, pathways)

    figure = plt.figure(figsize=(FIGURE_WIDTH, 198 / 25.4))
    rows = figure.add_gridspec(3, 1, height_ratios=(0.29, 0.38, 0.33), left=0.072, right=0.950, bottom=0.045, top=0.982, hspace=0.24)
    _workflow_panel(figure, rows[0], directories["mit"] / "workflow_example.npz")
    _heatmaps(figure, rows[1], summary)
    bottom = rows[2].subgridspec(1, 2, width_ratios=(0.37, 0.63), wspace=0.22)
    _linkage_panel(figure, bottom[0], linkage)
    _phenotype_panel(figure, bottom[1], phenotype)
    for label, spec in (("a", rows[0]), ("b", rows[1]), ("c", bottom[0]), ("d", bottom[1])):
        box = spec.get_position(figure)
        figure.text(box.x0 - 0.028, box.y1 + 0.006, f"({label})", fontsize=8, fontweight="bold", va="bottom")
    outputs = export_figure(figure, "population_individual")
    plt.close(figure)

    create_output_directories()
    summary.to_csv(AUDIT_ROOT / "population_individual_condition_summary.csv", index=False)
    linkage.to_csv(AUDIT_ROOT / "population_individual_soh_linkage.csv", index=False)
    pathways.to_csv(AUDIT_ROOT / "population_individual_failure_pathways.csv", index=False)
    return outputs
