"""R4-10/R4-11 canonical MIT L5 reconstruction-prognosis workflow.

The immutable MIT L5 waveform archives are aligned to the certified clean test
identity using their sample metrics table.  A clean population predictor and a
clean population-plus-individual predictor are fitted on train+validation
batteries, then evaluated on clean and reconstructed test representations.
No reconstruction model is retrained here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _certified as C
from . import _stats as S
from .future_prognosis_canonical import (
    HORIZON,
    TARGET,
    _feature_matrix,
    _fit_train_pca,
    _project,
    build_feature_table,
)
from .provenance import canonical_run
from analysis.utils.paths import REPOSITORY_ROOT

DATASET = "mit"
SEVERITY = "L5"
L5_ASSET_ROOT = REPOSITORY_ROOT / "analysis/assets/reconstruction/mit_l5"
CONDITIONS = {
    "downsample": "downsample",
    "random_missing": "random_missing",
    "part_missing": "part_missing",
    "destructive": "destructive",
    "hybrid": "hybrid",
}


def _flatten_individual_waveform(values: np.ndarray) -> np.ndarray:
    """Return one flattened individual waveform per eligible test row.

    The certified evaluator keeps waveforms in ``(N, 1000, 1)`` form for the
    decomposition metrics.  The longitudinal PCA is fitted on ``(N, 1000)``
    rows.  Flatten only the waveform
    representation at this boundary; do not alter the certified metrics.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"individual waveform must be at least 2-D, got {values.shape}")
    return values.reshape(values.shape[0], -1)


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray):
    """Fit the same scaled ridge convention as the registered prediction head."""
    x = np.asarray(train_x, float)
    y = np.asarray(train_y, float).reshape(-1)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    scaled = (x - mu) / sd
    y_mean = float(y.mean())
    alpha = float(C.RIDGE_ALPHA)
    coefficients = np.linalg.solve(
        scaled.T @ scaled + alpha * np.eye(scaled.shape[1]),
        scaled.T @ (y - y_mean),
    )

    def predict(values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, float) - mu) / sd) @ coefficients + y_mean

    return predict


def _load_l5(condition: str) -> tuple[np.ndarray, pd.DataFrame, Path, dict]:
    root = L5_ASSET_ROOT / condition / "level_5"
    waveform_path = root / "waveforms.npz"
    metrics_path = root / "sample_metrics.csv.gz"
    complete_path = root / "complete.json"
    if not waveform_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"missing immutable L5 archive for {condition}: {root}")
    with np.load(waveform_path, allow_pickle=False) as archive:
        reconstructed = np.asarray(archive["V_reconstructed"], dtype=np.float32)
    metrics = pd.read_csv(metrics_path).sort_values("sample_index", kind="stable")
    if len(metrics) != len(reconstructed):
        raise ValueError(f"archive/sample_metrics length mismatch for {condition}")
    complete = json.loads(complete_path.read_text()) if complete_path.is_file() else {}
    return reconstructed, metrics, waveform_path, complete


def _archive_order(metrics: pd.DataFrame) -> np.ndarray:
    return np.asarray([f"{battery}/{int(cycle)}.npz"
                       for battery, cycle in zip(metrics.battery, metrics.cycle)])


def _align_archive(reconstructed: np.ndarray, metrics: pd.DataFrame,
                   test: dict) -> np.ndarray:
    """Align immutable archive rows and restore the certified waveform shape.

    The L5 archive stores ``V_reconstructed`` as ``(N, 1000)`` while the
    certified evaluator consumes the repository's ``(N, 1000, 1)`` voltage
    contract.  Identity alignment happens in flattened row space; the shape
    conversion is restored at this boundary before calling the evaluator.
    """
    archive_keys = _archive_order(metrics)
    test_keys = np.asarray(test["keys"]).astype(str)
    if set(archive_keys) != set(test_keys):
        raise ValueError("immutable L5 archive identity set does not match certified MIT test set")
    order = {key: index for index, key in enumerate(archive_keys)}
    aligned = reconstructed[np.asarray([order[key] for key in test_keys])]
    expected_shape = np.asarray(test["voltage"]).shape
    if aligned.size != int(np.prod(expected_shape)):
        raise ValueError(
            f"immutable L5 waveform size {aligned.size} does not match "
            f"certified test shape {expected_shape}"
        )
    return aligned.reshape(expected_shape)


def _bootstrap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in frame[frame.representation_source == "reconstructed"].groupby(
            "corruption", sort=True):
        clean = frame[(frame.corruption == condition) &
                      (frame.representation_source == "clean")]
        joined = group.merge(clean, on=["battery_id", "cycle_id"], suffixes=("_recon", "_clean"))
        if joined.empty:
            continue
        for name, recon_column, clean_column in (
                ("population_fidelity_loss", "E_pop_recon", "E_pop_clean"),
                ("individual_fidelity_loss", "E_ind_recon", "E_ind_clean"),
                ("population_prognostic_loss", "population_abs_error_recon", "population_abs_error_clean"),
                ("individual_prognostic_loss", "individual_abs_error_recon", "individual_abs_error_clean")):
            values = joined[recon_column].to_numpy(float) - joined[clean_column].to_numpy(float)
            batteries = joined.battery_id.to_numpy()
            observed, low, high = S.battery_bootstrap(
                values, batteries, np.median, n=2000, seed=20260926)
            rows.append({"corruption": condition, "metric": name, "median": observed,
                         "bootstrap_low": low, "bootstrap_high": high,
                         "n_batteries": int(joined.battery_id.nunique())})
    return pd.DataFrame(rows)


def _smoke(run_id: str | None = None) -> Path:
    run = canonical_run("r4_reconstruction_prognosis", run_id)
    rows = []
    for condition in CONDITIONS.values():
        for cell in ("smoke1", "smoke2", "smoke3"):
            for cycle in (10, 20, 30):
                target = -0.01
                rows.extend([
                    {"battery_id": cell, "cycle_id": cycle, "corruption": condition,
                     "severity": SEVERITY, "representation_source": source,
                     "E_pop": 0.01 if source == "clean" else 0.02,
                     "E_ind": 0.0 if source == "clean" else 0.2,
                     "C_ind": 1.0 if source == "clean" else 0.8,
                     TARGET: target,
                     "population_prediction": target,
                     "individual_prediction": target,
                     "population_abs_error": 0.0 if source == "clean" else 0.002,
                     "individual_abs_error": 0.0 if source == "clean" else 0.001}
                    for source in ("clean", "reconstructed")])
    raw = pd.DataFrame(rows)
    run.write_table("raw", "reconstruction_prognosis_raw.csv.gz", raw)
    run.write_table("analysis", "reconstruction_prognosis_summary.csv", _bootstrap_summary(raw))
    run.write_table("plotting", "reconstruction_prognosis_plotting.csv", _bootstrap_summary(raw))
    config = {"dataset": DATASET, "severity": SEVERITY, "horizon": HORIZON,
              "status": "SMOKE"}
    run.write_json("reconstruction_prognosis_config.json", config)
    run.write_manifest(config=config, input_paths={
        "l5_asset_root": str(L5_ASSET_ROOT.relative_to(REPOSITORY_ROOT))})
    print(f"R4 reconstruction prognosis outputs -> {run.root}")
    return run.root


def run(run_id: str | None = None, smoke: bool = False) -> Path:
    if smoke:
        return _smoke(run_id)
    run = canonical_run("r4_reconstruction_prognosis", run_id)
    certified = C.module()
    train = certified.load_clean(DATASET, "train")
    validation = certified.load_clean(DATASET, "vali")
    test = certified.load_clean(DATASET, "test")
    spline, manifold, s_ind, _, _ = certified.select_manifold(train, validation)
    state = certified.prepare_test_decomposition(test, spline, manifold)
    sequences = C.load_sequences(DATASET, ("train", "vali", "test"))
    pca = _fit_train_pca(sequences["train"])
    tables = {split: build_feature_table(sequences, split, pca)
              for split in ("train", "vali", "test")}
    fit_pool = pd.concat([tables["train"], tables["vali"]], ignore_index=True)
    target_lookup = tables["test"].set_index(["battery_id", "cycle_id"])

    pop_train = _feature_matrix(fit_pool, "P")
    ind_train = _feature_matrix(fit_pool, "P+I")
    y_train = fit_pool[TARGET].to_numpy(float)
    train_ok = np.isfinite(y_train) & np.isfinite(pop_train).all(axis=1) & np.isfinite(ind_train).all(axis=1)
    population_predictor = _fit_ridge(pop_train[train_ok], y_train[train_ok])
    individual_predictor = _fit_ridge(ind_train[train_ok], y_train[train_ok])

    clean_metrics, clean_components = certified.evaluate_reconstruction(
        test["voltage"], test, state, spline, manifold, s_ind, train["soh"])
    eligible_keys = [(str(cell), int(cycle)) for cell, cycle in
                     zip(test["cells"][state["eligible"]], test["cycles"][state["eligible"]])]
    target_values = target_lookup.loc[eligible_keys, TARGET].to_numpy(float)
    future_ok = np.isfinite(target_values)
    if future_ok.sum() < 1:
        raise RuntimeError("R4 reconstruction prognosis has no exact-cycle future targets")
    eligible_keys = [key for key, keep in zip(eligible_keys, future_ok) if keep]
    clean_metrics = clean_metrics.iloc[future_ok].reset_index(drop=True)
    clean_components["rec_individual"] = clean_components["rec_individual"][future_ok]
    metric_index = {(str(row.cell), int(row.cycle)): row for _, row in clean_metrics.iterrows()}
    clean_pca = _project(
        pca, _flatten_individual_waveform(clean_components["rec_individual"])
    )
    clean_h = clean_metrics.h_rec.to_numpy(float)
    clean_pop_pred = population_predictor(clean_h.reshape(-1, 1))
    clean_ind_pred = individual_predictor(np.column_stack([clean_h, clean_pca]))

    all_rows = []
    for condition, asset_condition in CONDITIONS.items():
        reconstructed, sample_metrics, waveform_path, complete = _load_l5(asset_condition)
        aligned = _align_archive(reconstructed, sample_metrics, test)
        reconstructed_metrics, components = certified.evaluate_reconstruction(
            aligned, test, state, spline, manifold, s_ind, train["soh"])
        reconstructed_metrics = reconstructed_metrics.iloc[future_ok].reset_index(drop=True)
        components["rec_individual"] = components["rec_individual"][future_ok]
        recon_pca = _project(
            pca, _flatten_individual_waveform(components["rec_individual"])
        )
        recon_h = reconstructed_metrics.h_rec.to_numpy(float)
        recon_pop_pred = population_predictor(recon_h.reshape(-1, 1))
        recon_ind_pred = individual_predictor(np.column_stack([recon_h, recon_pca]))
        for index, key in enumerate(eligible_keys):
            target = target_lookup.loc[key, TARGET]
            clean_row = metric_index[key]
            recon_row = reconstructed_metrics.iloc[index]
            all_rows.extend([
                {"battery_id": key[0], "cycle_id": key[1], "corruption": condition,
                 "severity": SEVERITY, "representation_source": "clean",
                 "E_pop": float(clean_row.E_pop), "E_ind": float(clean_row.E_ind),
                 "C_ind": float(clean_row.C_ind), TARGET: float(target),
                 "population_prediction": float(clean_pop_pred[index]),
                 "individual_prediction": float(clean_ind_pred[index]),
                 "population_abs_error": abs(float(clean_pop_pred[index]) - float(target)),
                 "individual_abs_error": abs(float(clean_ind_pred[index]) - float(target))},
                {"battery_id": key[0], "cycle_id": key[1], "corruption": condition,
                 "severity": SEVERITY, "representation_source": "reconstructed",
                 "E_pop": float(recon_row.E_pop), "E_ind": float(recon_row.E_ind),
                 "C_ind": float(recon_row.C_ind), TARGET: float(target),
                 "population_prediction": float(recon_pop_pred[index]),
                 "individual_prediction": float(recon_ind_pred[index]),
                 "population_abs_error": abs(float(recon_pop_pred[index]) - float(target)),
                 "individual_abs_error": abs(float(recon_ind_pred[index]) - float(target))},
            ])
        C.log(f"    {condition}: aligned {len(reconstructed)} L5 waveforms")

    raw = pd.DataFrame(all_rows)
    summary = _bootstrap_summary(raw)
    run.write_table("raw", "reconstruction_prognosis_raw.csv.gz", raw)
    run.write_table("analysis", "reconstruction_prognosis_summary.csv", summary)
    run.write_table("plotting", "reconstruction_prognosis_plotting.csv", summary)
    config = {"dataset": DATASET, "severity": SEVERITY, "horizon": HORIZON,
              "target": "dSOH_100", "fit_split": ["train", "vali"],
              "evaluation_split": "test",
              "exact_future_cycle": True, "missing_future_cycle": "excluded",
              "conditions": list(CONDITIONS), "status": "COMPLETE"}
    run.write_json("reconstruction_prognosis_config.json", config)
    inventory = {
        condition: str((L5_ASSET_ROOT / asset_condition / "level_5" / "waveforms.npz")
                       .relative_to(REPOSITORY_ROOT))
        for condition, asset_condition in CONDITIONS.items()
    }
    run.write_json("l5_asset_inventory.json", inventory)
    run.write_manifest(config=config, input_paths={
        condition: str((L5_ASSET_ROOT / asset_condition / "level_5" / "waveforms.npz")
                       .relative_to(REPOSITORY_ROOT))
        for condition, asset_condition in CONDITIONS.items()}, seeds=[3407])
    print(f"R4 reconstruction prognosis outputs -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    command.add_argument("--smoke", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.run_id, args.smoke)


if __name__ == "__main__":
    main()
