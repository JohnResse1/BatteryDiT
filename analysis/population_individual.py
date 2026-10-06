"""Population/individual recoverability statistics across the six systems.

Reads the formal `PopulationIndividual` runs and writes:

    population_individual_condition_summary.csv   condition-level medians
    population_individual_soh_linkage.csv         rho(E_ind, SOH error) per system
    population_individual_manifold_validation.csv  manifold-vs-baseline validation
    population_individual_signature_linkage.csv    rho(C_ind, signature error) per system

No figure is produced here and no matplotlib is imported.
"""
from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.utils.io import analysis_dir, write_table
from analysis.utils.paths import RESULTS_ROOT

BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260829

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


def run_directory(dataset: str, seed: int, results_root: Path = RESULTS_ROOT) -> Path:
    """The single complete formal run for one system and model seed."""
    root = Path(results_root) / "PopulationIndividual" / dataset
    matches = []
    if root.is_dir():
        for summary_path in sorted(root.glob("*/summary.json")):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(payload.get("model_seed", -1)) == seed and payload.get("status") == "COMPLETE":
                matches.append(summary_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{dataset} population-individual seed {seed} requires exactly one "
            f"complete run; found {matches}")
    return matches[0]


def load_systems(results_root: Path = RESULTS_ROOT):
    """Concatenated battery-level and condition-level tables for all systems."""
    battery_frames, summary_frames = [], []
    for dataset, (label, seed) in SYSTEMS.items():
        directory = run_directory(dataset, seed, results_root)
        battery = pd.read_csv(directory / "battery_level_decomposition.csv")
        summary = pd.read_csv(directory / "condition_summary.csv")
        if set(battery.condition) != set(CONDITIONS) or set(summary.condition) != set(CONDITIONS):
            raise ValueError(f"{directory} has incomplete corruption topology coverage")
        if set(battery.severity) != set(LEVELS) or set(summary.severity) != set(LEVELS):
            raise ValueError(f"{directory} has incomplete severity coverage")
        if not battery.groupby("cell").size().eq(25).all() or len(summary) != 25:
            raise ValueError(f"{directory} does not contain 25 conditions per battery")
        battery.insert(0, "dataset", label)
        summary.insert(0, "dataset", label)
        battery_frames.append(battery)
        summary_frames.append(summary)
    return (pd.concat(battery_frames, ignore_index=True),
            pd.concat(summary_frames, ignore_index=True))


def _cluster_bootstrap_rho(frame: pd.DataFrame, seed: int):
    subset = frame[["cell", "E_ind", "soh_error"]].dropna()
    observed = float(spearmanr(subset.E_ind, subset.soh_error).statistic)
    groups = [group[["E_ind", "soh_error"]].to_numpy(float)
              for _, group in subset.groupby(subset.cell.astype(str), sort=True)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        draw = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[i] for i in draw])
        rho = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        raise RuntimeError("whole-battery Spearman bootstrap produced no finite value")
    low, high = np.percentile(values, (2.5, 97.5))
    return observed, float(low), float(high), len(values)


def soh_linkage(battery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(value[0] for value in SYSTEMS.values()):
        group = battery[battery.dataset == label]
        rho, low, high, finite = _cluster_bootstrap_rho(group, BOOTSTRAP_SEED + index)
        rows.append({"dataset": label, "spearman_rho": rho,
                     "bootstrap_low": low, "bootstrap_high": high,
                     "bootstrap_resamples": BOOTSTRAPS,
                     "finite_bootstrap_resamples": finite,
                     "n_batteries": int(group.cell.nunique()),
                     "n_battery_conditions": int(len(group))})
    return pd.DataFrame(rows)


def manifold_validation(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Per-system validation RMSE of the manifold vs the constant baseline."""
    rows = []
    for dataset, (label, seed) in SYSTEMS.items():
        directory = run_directory(dataset, seed, results_root)
        table = pd.read_csv(directory / "population_manifold_validation.csv")
        batteries = table[table.row_type == "selected_battery"].copy()
        if batteries.empty:
            raise ValueError(f"no per-battery manifold validation rows in {directory}")
        rows.append({
            "dataset": label,
            "constant_rmse": float(batteries.constant_rmse.mean()),
            "manifold_rmse": float(batteries.selected_rmse.mean()),
            "improved_batteries": int((batteries.selected_rmse < batteries.constant_rmse).sum()),
            "validation_batteries": int(len(batteries)),
        })
    return pd.DataFrame(rows)


def _bootstrap_signature(battery: pd.DataFrame, seed: int):
    observed = float(spearmanr(battery.C_ind, battery.signature_error).statistic)
    groups = [group[["C_ind", "signature_error"]].to_numpy(float)
              for _, group in battery.groupby("cell", sort=True)]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAPS):
        draw = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[i] for i in draw])
        value = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if np.isfinite(value):
            estimates.append(value)
    low, high = np.percentile(estimates, (2.5, 97.5))
    return observed, float(low), float(high), len(estimates)


def signature_linkage(battery: pd.DataFrame) -> pd.DataFrame:
    """rho(C_ind, signature error) per system with a whole-battery bootstrap CI."""
    rows = []
    for index, label in enumerate(value[0] for value in SYSTEMS.values()):
        group = battery[battery.dataset == label]
        rho, low, high, finite = _bootstrap_signature(group, 20260829 + 1000 * index)
        rows.append({"dataset": label, "spearman_rho": rho,
                     "bootstrap_low": low, "bootstrap_high": high,
                     "finite_bootstraps": finite,
                     "n_batteries": int(group.cell.nunique())})
    return pd.DataFrame(rows)


def export_workflow_example(directory: Path) -> Path:
    """Copy the frozen MIT workflow example so the figure layer never reads results/."""
    source = directory / "workflow_example.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    target = analysis_dir() / "population_individual_workflow_example.npz"
    shutil.copyfile(source, target)
    return target


def main() -> None:
    battery, summary = load_systems()
    write_table(summary, "population_individual_condition_summary.csv")
    write_table(soh_linkage(battery), "population_individual_soh_linkage.csv")
    write_table(manifold_validation(), "population_individual_manifold_validation.csv")
    write_table(signature_linkage(battery), "population_individual_signature_linkage.csv")
    export_workflow_example(run_directory("mit", SYSTEMS["mit"][1]))
    print(f"wrote 4 tables -> {analysis_dir()}")


if __name__ == "__main__":
    main()
