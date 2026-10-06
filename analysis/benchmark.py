"""Benchmark statistics for the heterogeneous-corruption figure.

Reads the formal `Reconstruction` / `PredictDownstream` / `DirectPrediction`
runs and writes:

    benchmark_summary.csv            mean/sd per model x condition, recon + assisted + direct
    benchmark_soh_ci.csv             whole-battery CI for BatteryDiT->TRB SOH error
    benchmark_representatives.npz    representative clean/observed/recon/baseline curves

No matplotlib is imported here.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.utils.io import analysis_dir, write_table
from analysis.utils.paths import RESULTS_ROOT
from analysis.utils.results import (FORMAL_CONDITIONS, condition_matrix, mean_and_std,
                            require_runs, select_runs)

DEGRADED_CONDITIONS = ("downsample", "random", "segment", "destructive")
RECONSTRUCTION_MODELS = OrderedDict([
    ("batterydit", "BatteryDiT"), ("csdi", "CSDI"),
    ("transformer", "Transformer"), ("kalman", "Kalman"),
])
DIRECT_MODELS = OrderedDict([
    ("gru", "GRU"), ("lstm", "LSTM"), ("transformer", "Transformer"),
    ("tcn", "TCN"), ("trb", "TRB"),
])


def formal_runs(dataset: str = "mit", results_root: Path = RESULTS_ROOT):
    reconstruction, assisted, direct = OrderedDict(), OrderedDict(), OrderedDict()
    for model, label in RECONSTRUCTION_MODELS.items():
        reconstruction[model] = require_runs(
            select_runs("Reconstruction", dataset, recon_model=model,
                        results_root=results_root),
            f"{label} reconstruction", 1 if model == "kalman" else 3)
        assisted[model] = require_runs(
            select_runs("PredictDownstream", dataset, recon_model=model,
                        predictor="trb", results_root=results_root),
            f"{label}->TRB", 3)
    for model, label in DIRECT_MODELS.items():
        direct[model] = require_runs(
            select_runs("DirectPrediction", dataset, direct_model=model,
                        results_root=results_root),
            f"direct {label}", 3)
    return reconstruction, assisted, direct


def _seed_run(runs, seed: int = 2021):
    matching = [run for run in runs if run.seed == seed]
    if len(matching) != 1:
        raise FileNotFoundError(f"expected one seed {seed} run, found {[r.seed for r in runs]}")
    return matching[0]


def _trajectory(run, condition: str) -> dict:
    path = run.directory / f"reconstructions_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing full trajectory output: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"input_voltage", "clean_voltage", "reconstruction", "keys"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def representatives(reconstruction_runs, reconstruction_summary):
    """One median-error cycle per degraded condition, plus the best baseline."""
    selected = {}
    batterydit = _seed_run(reconstruction_runs["batterydit"])
    for condition in DEGRADED_CONDITIONS:
        ours = _trajectory(batterydit, condition)
        errors = np.sqrt(np.mean((ours["reconstruction"] - ours["clean_voltage"]) ** 2, axis=(1, 2)))
        index = int(np.argmin(np.abs(errors - np.median(errors))))
        key = str(ours["keys"][index])
        condition_index = FORMAL_CONDITIONS.index(condition)
        baseline_model = min(
            (m for m in RECONSTRUCTION_MODELS if m != "batterydit"),
            key=lambda m: reconstruction_summary[m][0][condition_index])
        baseline = _trajectory(_seed_run(reconstruction_runs[baseline_model]), condition)
        lookup = {str(v): i for i, v in enumerate(baseline["keys"])}
        if key not in lookup:
            raise ValueError(f"representative key {key} absent from {baseline_model}/{condition}")
        selected[condition] = {
            "clean": ours["clean_voltage"][index].reshape(-1),
            "observed": ours["input_voltage"][index].reshape(-1),
            "batterydit": ours["reconstruction"][index].reshape(-1),
            "baseline": baseline["reconstruction"][lookup[key]].reshape(-1),
        }
    return selected


def soh_error_ci(assisted_runs, reconstruction_runs):
    """Whole-battery bootstrap CI of the BatteryDiT->TRB absolute SOH error."""
    prediction_run = _seed_run(assisted_runs["batterydit"])
    reconstruction_run = _seed_run(reconstruction_runs["batterydit"])
    path = prediction_run.directory / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing prediction arrays: {path}")
    rows = []
    with np.load(path, allow_pickle=False) as archive:
        for condition in FORMAL_CONDITIONS:
            prediction = np.asarray(archive[f"{condition}_prediction"])[:, 0]
            target = np.asarray(archive[f"{condition}_target"])[:, 0]
            key_name = f"{condition}_keys"
            keys = (np.asarray(archive[key_name]).astype(str)
                    if key_name in archive.files
                    else _trajectory(reconstruction_run, condition)["keys"].astype(str))
            errors = np.abs(prediction - target)
            cells = np.asarray([key.split("/", 1)[0] for key in keys])
            cell_values = np.asarray([errors[cells == cell].mean()
                                      for cell in sorted(set(cells))])
            rng = np.random.default_rng(20260829)
            draws = rng.integers(0, len(cell_values), size=(10_000, len(cell_values)))
            bootstrap = cell_values[draws].mean(axis=1)
            low, high = np.quantile(bootstrap, (0.025, 0.975))
            rows.append({"condition": condition, "soh_mae": float(errors.mean()),
                         "ci_low": float(low), "ci_high": float(high),
                         "n_batteries": len(cell_values)})
    return pd.DataFrame(rows)


def cycle_errors(assisted_runs, reconstruction_runs):
    """Per-cycle absolute SOH error for BatteryDiT->TRB, keyed by condition."""
    prediction_run = _seed_run(assisted_runs["batterydit"])
    path = prediction_run.directory / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing prediction arrays: {path}")
    output = {}
    with np.load(path, allow_pickle=False) as archive:
        for condition in FORMAL_CONDITIONS:
            prediction = np.asarray(archive[f"{condition}_prediction"])[:, 0]
            target = np.asarray(archive[f"{condition}_target"])[:, 0]
            output[condition] = np.abs(prediction - target)
    return output


def main() -> None:
    dataset = "mit"
    reconstruction, assisted, direct = formal_runs(dataset)
    rows = []
    summaries = {}
    for kind, runs, metric in (("reconstruction", reconstruction, "battery_balanced_rmse"),
                               ("assisted", assisted, "soh_rmse"),
                               ("direct", direct, "soh_rmse")):
        summaries[kind] = OrderedDict((model, mean_and_std(condition_matrix(items, metric)))
                                      for model, items in runs.items())
        for model, (mean, deviation) in summaries[kind].items():
            for ci, condition in enumerate(FORMAL_CONDITIONS):
                rows.append({"task": kind, "model": model, "condition": condition,
                             "value_mean": float(mean[ci]), "value_sd": float(deviation[ci])})
    write_table(pd.DataFrame(rows), "benchmark_summary.csv")
    write_table(soh_error_ci(assisted, reconstruction), "benchmark_soh_ci.csv")
    waveform_rows = []
    for model, (mean, deviation) in summaries["reconstruction"].items():
        for ci, condition in enumerate(FORMAL_CONDITIONS):
            waveform_rows.append({"model": model, "condition": condition,
                                  "value_mean": float(mean[ci]), "value_sd": float(deviation[ci])})
    write_table(pd.DataFrame(waveform_rows), "benchmark_waveform_rmse.csv")
    errors = cycle_errors(assisted, reconstruction)
    write_table(pd.DataFrame({
        "condition": [c for c in FORMAL_CONDITIONS for _ in errors[c]],
        "cycle_error": np.concatenate([errors[c] for c in FORMAL_CONDITIONS]),
    }), "benchmark_cycle_errors.csv")
    np.savez_compressed(analysis_dir() / "benchmark_representatives.npz",
                        **{f"{condition}_{name}": values
                           for condition, curves in representatives(reconstruction,
                                                                    summaries["reconstruction"]).items()
                           for name, values in curves.items()})
    print(f"wrote benchmark tables -> {analysis_dir()}")


if __name__ == "__main__":
    main()
