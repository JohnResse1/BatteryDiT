"""Cross-system trajectory recovery and downstream utility statistics.

Reads the formal `multi_dataset` runs and writes:

    multi_dataset_summary.csv          per system x condition reconstruction + prediction delta
    multi_dataset_corrupted_rmse.csv   realised corrupted-input RMSE per system x condition
    multi_dataset_representatives.npz  one locked cycle per system and
                                      condition

No matplotlib is imported here.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.utils.io import analysis_dir, write_table
from analysis.utils.paths import RESULTS_ROOT
from analysis.utils.results import (FORMAL_CONDITIONS, condition_matrix, require_runs,
                            select_runs)

DATASETS = OrderedDict([
    ("mit", "MIT"), ("hust", "HUST"), ("mich", "MICH"), ("sdu", "SDU"),
    ("stanford_2", "Stanford"), ("tju_nca_45c", "TJU-NCA"), ("tju_ncm_45c", "TJU-NCM"),
])
CHEMISTRIES = {
    "mit": "LFP", "hust": "LFP", "mich": "NMC111", "sdu": "NMC532",
    "stanford_2": "NMC532", "tju_nca_45c": "NCA", "tju_ncm_45c": "NCM",
}
DEGRADED_CONDITIONS = OrderedDict([
    ("downsample", "Downsample"), ("random", "Random"),
    ("segment", "Segment"), ("destructive", "Destructive"),
])
SEED_FOR_INPUT = 2021
MAX_CANDIDATES_PER_BATTERY = 64


def runs(dataset: str, results_root: Path = RESULTS_ROOT):
    """Formal runs for one system.

    The six external systems live under `results/multi_dataset/`; MIT is the
    in-domain reference and lives under the top-level `results/` groups, so it
    is resolved from there.
    """
    if dataset == "mit":
        reconstruction = require_runs(
            select_runs("Reconstruction", dataset, recon_model="batterydit",
                        results_root=results_root),
            "MIT reconstruction", 3)
        prediction = require_runs(
            select_runs("PredictDownstream", dataset, recon_model="batterydit",
                        predictor="trb", results_root=results_root),
            "MIT downstream prediction", 3)
        return reconstruction, prediction
    reconstruction = require_runs(
        select_runs("multi_dataset", dataset, recon_model="batterydit",
                    task_name="Reconstruction", results_root=results_root),
        f"{dataset} multi-dataset reconstruction", 3)
    prediction = require_runs(
        select_runs("multi_dataset", dataset, recon_model="batterydit", predictor="trb",
                    task_name="PredictDownstream", results_root=results_root),
        f"{dataset} multi-dataset downstream prediction", 3)
    return reconstruction, prediction


def _trajectory_path(run, condition: str) -> Path:
    path = run.directory / f"reconstructions_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing full trajectory output: {path}")
    return path


def _candidate_indices(cells: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for cell in dict.fromkeys(cells.astype(str)):
        members = np.flatnonzero(cells == cell)
        count = min(len(members), MAX_CANDIDATES_PER_BATTERY)
        positions = np.unique(np.rint(np.linspace(0, len(members) - 1, count)).astype(int))
        selected.extend(members[positions].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def representative(recon_runs):
    """One cycle chosen by a battery-balanced median rule.

    The selection is made on the segment-missing view, then the same locked
    cycle is exported for every formal corruption condition.  Reconstructions
    are averaged over the three formal model seeds; the seed-specific RMSEs
    remain in the metadata for annotations in trajectory figures.
    """
    condition = "segment"
    first = _trajectory_path(next(run for run in recon_runs if run.seed == SEED_FOR_INPUT), condition)
    with np.load(first, allow_pickle=False) as archive:
        keys = np.asarray(archive["keys"]).astype(str)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{first} contains duplicate sample keys")
    cells = np.asarray([key.split("/", 1)[0] for key in keys])
    candidates = _candidate_indices(cells)
    seed_errors, candidate_reconstructions = [], []
    candidate_clean = candidate_input = None
    for run in recon_runs:
        path = _trajectory_path(run, condition)
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(np.asarray(archive["keys"]).astype(str), keys):
                raise ValueError(f"trajectory key order differs across seeds: {path}")
            clean = np.asarray(archive["clean_voltage"], dtype=np.float32)
            reconstruction = np.asarray(archive["reconstruction"], dtype=np.float32)
            seed_errors.append(np.sqrt(np.mean((reconstruction - clean) ** 2, axis=(1, 2)))[candidates])
            candidate_reconstructions.append(reconstruction[candidates, :, 0])
            if run.seed == SEED_FOR_INPUT:
                candidate_clean = clean[candidates, :, 0]
                candidate_input = np.asarray(archive["input_voltage"], dtype=np.float32)[candidates, :, 0]
    if candidate_clean is None or candidate_input is None:
        raise FileNotFoundError(f"seed {SEED_FOR_INPUT} trajectory is required")
    score = np.stack(seed_errors).mean(axis=0)
    local = int(np.argmin(np.abs(score - np.median(score))))
    selected_index = int(candidates[local])
    selected_key = str(keys[selected_index])
    item = {
        "clean": candidate_clean[local],
        "input": candidate_input[local],
        "reconstruction": np.stack(candidate_reconstructions).mean(axis=0)[local],
        "observed": np.abs(candidate_input[local]) > 1e-12,
    }
    condition_rmse = {}
    for condition in DEGRADED_CONDITIONS:
        seed_reconstructions = []
        seed_errors = []
        clean_curve = input_curve = None
        for run in recon_runs:
            path = _trajectory_path(run, condition)
            with np.load(path, allow_pickle=False) as archive:
                archive_keys = np.asarray(archive["keys"]).astype(str)
                if not np.array_equal(archive_keys, keys):
                    raise ValueError(f"trajectory key order differs across seeds: {path}")
                clean = np.asarray(archive["clean_voltage"], dtype=np.float32)
                input_voltage = np.asarray(archive["input_voltage"], dtype=np.float32)
                reconstruction = np.asarray(archive["reconstruction"], dtype=np.float32)
                if clean_curve is None:
                    clean_curve = clean[selected_index, :, 0].copy()
                elif not np.allclose(clean_curve, clean[selected_index, :, 0]):
                    raise ValueError(f"clean trajectory differs across seeds: {path}")
                if run.seed == SEED_FOR_INPUT:
                    input_curve = input_voltage[selected_index, :, 0].copy()
                curve = reconstruction[selected_index, :, 0].copy()
                seed_reconstructions.append(curve)
                seed_errors.append(float(np.sqrt(np.mean((curve - clean_curve) ** 2))))
        if clean_curve is None or input_curve is None:
            raise FileNotFoundError(f"seed {SEED_FOR_INPUT} trajectory is required")
        item[f"{condition}_clean"] = clean_curve
        item[f"{condition}_input"] = input_curve
        item[f"{condition}_reconstruction"] = np.mean(
            np.stack(seed_reconstructions), axis=0
        )
        item[f"{condition}_observed"] = np.abs(input_curve) > 1e-12
        condition_rmse[condition] = {
            "mean": float(np.mean(seed_errors)),
            "sd": float(np.std(seed_errors, ddof=1)),
        }
    metadata = {
        "key": selected_key,
        "selected_index": selected_index,
        "candidate_count": int(len(candidates)),
        "selection_score_v": float(score[local]),
        "candidate_median_score_v": float(np.median(score)),
        "condition_rmse_v": condition_rmse,
        "selection_rule": "up to 64 evenly spaced cycles per battery; nearest to the "
                           "median three-seed mean waveform RMSE on Segment missing",
    }
    return item, metadata


def main() -> None:
    summary_rows, corrupted_rows, payload, metadata = [], [], {}, {}
    columns = [FORMAL_CONDITIONS.index(name) for name in DEGRADED_CONDITIONS]
    for dataset, label in DATASETS.items():
        recon_runs, prediction_runs = runs(dataset)
        recon_values = condition_matrix(recon_runs, "battery_balanced_rmse")[:, columns]
        prediction_values = condition_matrix(prediction_runs, "soh_rmse")
        clean_index = FORMAL_CONDITIONS.index("clean")
        deltas = prediction_values[:, columns] - prediction_values[:, [clean_index]]
        for ci, condition in enumerate(DEGRADED_CONDITIONS):
            summary_rows.append({
                "dataset": label, "chemistry": CHEMISTRIES[dataset], "condition": condition,
                "reconstruction_mean": float(recon_values.mean(axis=0)[ci]),
                "reconstruction_sd": float(recon_values.std(axis=0, ddof=1)[ci]),
                "prediction_delta_mean": float(deltas.mean(axis=0)[ci]),
                "prediction_delta_sd": float(deltas.std(axis=0, ddof=1)[ci]),
                "n_model_seeds": int(recon_values.shape[0])})
        input_run = next(run for run in recon_runs if run.seed == SEED_FOR_INPUT)
        for condition in DEGRADED_CONDITIONS:
            with np.load(_trajectory_path(input_run, condition), allow_pickle=False) as archive:
                corrupted = np.asarray(archive["input_voltage"], dtype=np.float32)
                clean = np.asarray(archive["clean_voltage"], dtype=np.float32)
            corrupted_rows.append({"dataset": label, "condition": condition,
                                   "corrupted_rmse": float(np.sqrt(np.mean((corrupted - clean) ** 2,
                                                                           axis=(1, 2))).mean())})
        item, meta = representative(recon_runs)
        payload.update({f"{dataset}_{name}": value for name, value in item.items()})
        metadata[dataset] = meta
    write_table(pd.DataFrame(summary_rows), "multi_dataset_summary.csv")
    write_table(pd.DataFrame(corrupted_rows), "multi_dataset_corrupted_rmse.csv")
    np.savez_compressed(analysis_dir() / "multi_dataset_representatives.npz", **payload)
    (analysis_dir() / "multi_dataset_representatives.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote multi-dataset tables -> {analysis_dir()}")


if __name__ == "__main__":
    main()
