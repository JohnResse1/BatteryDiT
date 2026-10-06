"""Stress-test statistics for the four reconstruction models.

Reads the formal `ReconstructionStressTest` / `PredictStressTest` /
`DegradationSignature` runs and writes:

    stress_test_curves.csv          mean/sd curves per model x corruption x level
    stress_test_statistics.json     signature concordance and fit statistics
    stress_test_signature_raw.csv   per-seed clean/corrupted/reconstructed tau

No matplotlib is imported here.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from analysis.utils.io import analysis_dir, write_table
from analysis.utils.paths import RESULTS_ROOT

LEVELS = np.arange(1, 6)
CONDITIONS = OrderedDict([
    ("downsample", "Downsample"),
    ("random_missing", "Random missing"),
    ("segment_missing", "Segment missing"),
    ("destructive", "Destructive"),
    ("hybrid", "Hybrid"),
])
MODELS = OrderedDict([
    ("batterydit", "BatteryDiT"), ("csdi", "CSDI"),
    ("transformer", "Transformer"), ("kalman", "Kalman"),
])


def _payloads(group: str, dataset: str, filename: str, model: str,
              results_root: Path) -> list[dict]:
    root = results_root / group / dataset
    payloads = []
    if root.is_dir():
        for path in sorted(root.glob(f"*/{filename}")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("reconstruction_model", "")) == model:
                payloads.append(payload)
    return sorted(payloads, key=lambda item: int(item["model_seed"]))


def require(payloads: list[dict], label: str, expected: int) -> list[dict]:
    """Formal completeness check: exact models seeds and fixed evaluation seed."""
    if len(payloads) != expected:
        seeds = [item.get("model_seed") for item in payloads]
        raise FileNotFoundError(
            f"{label} requires {expected} stress result(s); found model seeds {seeds}")
    expected_seeds = [2021] if expected == 1 else [2021, 2022, 2023]
    if [int(item["model_seed"]) for item in payloads] != expected_seeds:
        raise ValueError(f"{label} model seeds are {[item['model_seed'] for item in payloads]}")
    if {int(item["evaluation_seed"]) for item in payloads} != {3407}:
        raise ValueError(f"{label} must use fixed evaluation seed 3407")
    return payloads


def curves(payloads: list[dict], metric: str):
    values = np.asarray([
        [[float(row[metric]) for row in payload[condition]] for condition in CONDITIONS]
        for payload in payloads])
    if values.shape[1:] != (5, 5) or not np.isfinite(values).all():
        raise ValueError(f"invalid stress matrix for {metric}: {values.shape}")
    deviation = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])
    return values.mean(axis=0), deviation, values


def load_stress(dataset: str = "mit", results_root: Path = RESULTS_ROOT):
    reconstruction, prediction = OrderedDict(), OrderedDict()
    for model, label in MODELS.items():
        reconstruction[label] = curves(
            require(_payloads("ReconstructionStressTest", dataset, "stress_metrics.json",
                              model, results_root),
                    f"{label} reconstruction", 1 if model == "kalman" else 3),
            "sample_rmse_mean")
        prediction[label] = curves(
            require(_payloads("PredictStressTest", dataset, "prediction_stress_metrics.json",
                              model, results_root), f"{label}->TRB", 3),
            "soh_rmse")
    return reconstruction, prediction


def load_signatures(dataset: str = "mit", results_root: Path = RESULTS_ROOT) -> list[dict]:
    paths = sorted((results_root / "DegradationSignature" / dataset).glob(
        "*/degradation_signature.npz"))
    payloads = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]) for name in archive.files}
        if tuple(payload["topologies"].astype(str)) != tuple(CONDITIONS):
            raise ValueError(f"topology order mismatch in {path}")
        payloads.append(payload)
    payloads.sort(key=lambda item: int(item["seed"]))
    if [int(item["seed"]) for item in payloads] != [2021, 2022, 2023]:
        raise FileNotFoundError("degradation signature requires model seeds 2021-2023")
    if {int(item["evaluation_seed"]) for item in payloads} != {3407}:
        raise ValueError("degradation signatures must use fixed evaluation seed 3407")
    return payloads


def main() -> None:
    dataset = "mit"
    reconstruction, prediction = load_stress(dataset)
    rows = []
    for kind, table in (("reconstruction", reconstruction), ("prediction", prediction)):
        for model, (mean, deviation, values) in table.items():
            for ci, condition in enumerate(CONDITIONS):
                for li, level in enumerate(LEVELS):
                    rows.append({"task": kind, "model": model,
                                 "condition": condition, "severity": f"L{int(level)}",
                                 "severity_index": int(level),
                                 "value_mean": float(mean[ci, li]),
                                 "value_sd": float(deviation[ci, li]),
                                 "n_model_seeds": int(values.shape[0])})
    write_table(pd.DataFrame(rows), "stress_test_curves.csv")

    signatures = load_signatures(dataset)
    reconstructed_tau = np.stack([item["reconstructed_tau"] for item in signatures])
    clean_tau = np.asarray([float(item["clean_tau"]) for item in signatures])
    corrupted_tau = np.stack([item["corrupted_tau"] for item in signatures])
    # x-axis of the relation panel: mean reconstructed degradation-order tau
    # over conditions x severity levels, paired with the BatteryDiT SOH RMSE
    tau_mean = reconstructed_tau.mean(axis=0)
    x_values = tau_mean.ravel()
    y_values = prediction["BatteryDiT"][0].ravel()
    rho, p_value = spearmanr(x_values, y_values)
    regression = linregress(x_values, y_values)
    write_table(pd.DataFrame({
        "condition": [c for c in CONDITIONS for _ in LEVELS],
        "severity": [f"L{int(l)}" for _ in CONDITIONS for l in LEVELS],
        "reconstructed_tau_mean": x_values,
        "batterydit_soh_rmse": y_values,
    }), "stress_test_signature_raw.csv")
    # figure-ready arrays (the figure layer must not touch `results/`)
    np.savez_compressed(
        analysis_dir() / "stress_test_figure_arrays.npz",
        levels=LEVELS,
        topologies=np.asarray(list(CONDITIONS)),
        models=np.asarray(list(MODELS.values())),
        reconstruction_mean=np.stack([reconstruction[m][0] for m in MODELS.values()]),
        reconstruction_sd=np.stack([reconstruction[m][1] for m in MODELS.values()]),
        prediction_mean=np.stack([prediction[m][0] for m in MODELS.values()]),
        prediction_sd=np.stack([prediction[m][1] for m in MODELS.values()]),
        clean_tau=clean_tau,
        corrupted_tau=corrupted_tau,
        reconstructed_tau=reconstructed_tau,
        representative_pca_clean=signatures[0]["representative_pca_clean"],
        representative_pca_corrupted=signatures[0]["representative_pca_corrupted"],
        representative_pca_reconstructed=signatures[0]["representative_pca_reconstructed"],
        representative_soh=signatures[0]["representative_soh"],
    )
    (analysis_dir() / "stress_test_statistics.json").write_text(json.dumps({
        "signature_statistics": {
            "relationship": {
                "spearman_rho": float(rho), "p_value": float(p_value),
                "slope": float(regression.slope), "intercept": float(regression.intercept),
                "r_value": float(regression.rvalue),
                "n_condition_levels": int(len(x_values)),
            },
            "clean_tau": clean_tau.tolist(),
            "reconstructed_tau": reconstructed_tau.tolist(),
            "topologies": list(CONDITIONS),
            "model_seeds": [int(item["seed"]) for item in signatures],
        },
        "metadata": {
            "dataset": dataset,
            "cycles_per_battery": int(signatures[0]["cycles_per_cell"]),
            "test_samples": int(len(signatures[0]["representative_soh"])),
        },
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote stress-test tables -> {analysis_dir()}")


if __name__ == "__main__":
    main()
