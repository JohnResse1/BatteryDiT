"""R4-06..R4-09 canonical future-prognosis workflow.

The primary target is the frozen exact-cycle target ``dSOH_100``.  Features
use only the current SOH and the preceding clean trajectory of the same
battery.  Predictors are fitted on train+validation batteries and evaluated
on held-out test batteries.  The workflow exports cycle-level predictions so
the paper layer never has to reconstruct a summary from model objects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import _certified as C
from . import _stats as S
from .heads import HEADS
from .provenance import canonical_run

TARGET = "dSOH_100"
HORIZON = 100
FEATURE_SETS = ("P", "P+I", "P+H", "P+H+I")
PREDICTORS = ("ridge", "knn", "mlp")
HISTORY_WINDOWS = (50, 100)


def _synthetic_sequences() -> dict:
    rng = np.random.default_rng(7)

    def make(split: str) -> list[dict]:
        out = []
        for index in range(6):
            cycle = np.arange(1, 181)
            soh = 0.99 - 0.00035 * cycle - 0.00001 * cycle ** 1.15
            residual = rng.normal(0, 0.02, (len(cycle), 24))
            out.append({"cell": f"smoke{split}{index}", "batch": "sm",
                        "cycle": cycle, "soh": soh, "residual": residual})
        return out

    return {split: make(split) for split in ("train", "vali", "test")}


def _fit_train_pca(sequences: list[dict]) -> dict:
    matrix = np.concatenate([sequence["residual"] for sequence in sequences], axis=0)
    mean = matrix.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(matrix - mean, full_matrices=False)
    return {"mean": mean, "components": components[:C.PRIMARY_K]}


def _project(pca: dict, residual: np.ndarray) -> np.ndarray:
    return (np.asarray(residual, float) - pca["mean"]) @ pca["components"].T


def _slope(values: np.ndarray, window: int) -> float:
    values = np.asarray(values, float)[-window:]
    if len(values) < 2:
        return float("nan")
    time = np.arange(len(values), dtype=float)
    centered = time - time.mean()
    denominator = float((centered ** 2).sum())
    return float((centered * (values - values.mean())).sum() / denominator)


def build_feature_table(sequences: dict, split: str, pca: dict,
                        horizon: int = HORIZON) -> pd.DataFrame:
    """Create one leakage-safe row per eligible battery/cycle."""
    rows = []
    n_components = pca["components"].shape[0]
    for sequence in sequences[split]:
        cycle = np.asarray(sequence["cycle"], dtype=int)
        soh = np.asarray(sequence["soh"], dtype=float)
        projected = _project(pca, sequence["residual"])
        for index, cycle_id in enumerate(cycle):
            future_index = int(np.searchsorted(cycle, cycle_id + horizon))
            if future_index >= len(cycle) or cycle[future_index] != cycle_id + horizon:
                target = np.nan
            else:
                target = float(soh[future_index] - soh[index])
            row = {
                "battery_id": str(sequence["cell"]),
                "batch": str(sequence.get("batch", str(sequence["cell"])[:2])),
                "split": split,
                "cycle_id": int(cycle_id),
                "soh": float(soh[index]),
                TARGET: target,
                "soh_slope_50": _slope(soh[:index + 1], HISTORY_WINDOWS[0]),
                "soh_slope_100": _slope(soh[:index + 1], HISTORY_WINDOWS[1]),
            }
            row.update({f"PC{component}": float(projected[index, component])
                       for component in range(n_components)})
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_matrix(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    population = frame[["soh"]].to_numpy(float)
    individual = frame[[column for column in frame.columns if column.startswith("PC")]].to_numpy(float)
    history = frame[["soh_slope_50", "soh_slope_100"]].to_numpy(float)
    blocks = {"P": population, "I": individual, "H": history}
    names = feature_set.replace("+", " ").split()
    return np.hstack([blocks[name] for name in names])


def _metric_rows(y_true: np.ndarray, y_pred: np.ndarray,
                 batteries: np.ndarray) -> dict:
    return {"mae": S.mae(y_true, y_pred), "r2": S.r2_score(y_true, y_pred),
            "spearman": S.spearman(y_true, y_pred), "n": int(len(y_true)),
            "n_batteries": int(len(set(batteries.astype(str))))}


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_set: str,
                 predictor: str) -> tuple[pd.DataFrame, dict]:
    x_train = _feature_matrix(train, feature_set)
    x_test = _feature_matrix(test, feature_set)
    y_train = train[TARGET].to_numpy(float)
    y_test = test[TARGET].to_numpy(float)
    train_ok = np.isfinite(y_train) & np.isfinite(x_train).all(axis=1)
    test_ok = np.isfinite(y_test) & np.isfinite(x_test).all(axis=1)
    if train_ok.sum() < 20 or test_ok.sum() < 10:
        return pd.DataFrame(), {"status": "INSUFFICIENT_EXACT_FUTURE_TARGETS"}
    prediction = HEADS[predictor](x_train[train_ok], y_train[train_ok], x_test[test_ok])
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    output = test.loc[test_ok, ["battery_id", "batch", "split", "cycle_id", TARGET]].copy()
    output["predictor"] = predictor
    output["feature_set"] = feature_set
    output["y_true"] = y_test[test_ok]
    output["y_pred"] = prediction
    output["absolute_error"] = np.abs(output.y_true - output.y_pred)
    return output, _metric_rows(output.y_true.to_numpy(), output.y_pred.to_numpy(),
                                output.battery_id.to_numpy()) | {"status": "READY"}


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, predictor), group in frame.groupby(["feature_set", "predictor"], sort=True):
        metrics = _metric_rows(group.y_true.to_numpy(float), group.y_pred.to_numpy(float),
                               group.battery_id.to_numpy())
        rows.append({"feature_set": feature_set, "predictor": predictor, **metrics})
    return pd.DataFrame(rows)


def _gains(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = (("P+I", "P", "R4-06_individual_gain"),
                   ("P+H+I", "P+H", "R4-08_history_individual_gain"))
    for predictor in PREDICTORS:
        for high, low, label in comparisons:
            high_frame = predictions[(predictions.predictor == predictor) &
                                     (predictions.feature_set == high)]
            low_frame = predictions[(predictions.predictor == predictor) &
                                    (predictions.feature_set == low)]
            if high_frame.empty or low_frame.empty:
                continue
            merged = high_frame.merge(low_frame, on=["battery_id", "cycle_id"],
                                      suffixes=("_high", "_low"))
            if merged.empty:
                continue
            delta, low_ci, high_ci = S.battery_bootstrap_delta_mae(
                merged.y_true_high.to_numpy(float), merged.y_pred_high.to_numpy(float),
                merged.y_pred_low.to_numpy(float), merged.battery_id.to_numpy(), n=1000)
            rows.append({"comparison": label, "predictor": predictor,
                         "feature_set_high": high, "feature_set_low": low,
                         "delta_mae_low_minus_high": delta,
                         "bootstrap_low": low_ci, "bootstrap_high": high_ci,
                         "n_batteries": int(merged.battery_id.nunique())})
    return pd.DataFrame(rows)


def _batch_controls(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (batch, feature_set, predictor), group in predictions.groupby(
            ["batch", "feature_set", "predictor"], sort=True):
        metrics = _metric_rows(group.y_true.to_numpy(float), group.y_pred.to_numpy(float),
                               group.battery_id.to_numpy())
        rows.append({"batch": batch, "feature_set": feature_set,
                     "predictor": predictor, **metrics})
    return pd.DataFrame(rows)


def run(dataset: str = "mit", run_id: str | None = None, smoke: bool = False) -> Path:
    run = canonical_run("r4_future_prognosis", run_id)
    sequences = _synthetic_sequences() if smoke else C.load_sequences(dataset, ("train", "vali", "test"))
    pca = _fit_train_pca(sequences["train"])
    tables = {split: build_feature_table(sequences, split, pca) for split in ("train", "vali", "test")}
    fit_pool = pd.concat([tables["train"], tables["vali"]], ignore_index=True)

    predictions = []
    for feature_set in FEATURE_SETS:
        for predictor in PREDICTORS:
            frame, _ = _fit_predict(fit_pool, tables["test"], feature_set, predictor)
            if not frame.empty:
                predictions.append(frame)
    if not predictions:
        raise RuntimeError("R4 prognosis produced no exact-cycle test targets")
    raw = pd.concat(predictions, ignore_index=True)
    run.write_table("raw", "future_prognosis_predictions.csv.gz", raw)
    run.write_table("analysis", "future_prognosis_summary.csv", _summary(raw))
    run.write_table("analysis", "future_prognosis_gains.csv", _gains(raw))
    run.write_table("analysis", "future_prognosis_batch_controls.csv", _batch_controls(raw))
    run.write_table("plotting", "future_prognosis_plotting.csv", _summary(raw))
    config = {"dataset": dataset, "target": TARGET, "horizon": HORIZON,
              "feature_sets": list(FEATURE_SETS), "predictors": list(PREDICTORS),
              "fit_split": ["train", "vali"], "evaluation_split": "test",
              "exact_future_cycle": True, "missing_future_cycle": "excluded",
              "status": "SMOKE" if smoke else "COMPLETE"}
    run.write_json("future_prognosis_config.json", config)
    run.write_manifest(config=config, input_paths={
        "clean_data": f"source_code/data/{dataset}/{{train,vali,test}}/clean"},
        seeds=[C.SYSTEMS[dataset][1]])
    print(f"R4 future prognosis outputs -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset", default="mit")
    command.add_argument("--run-id")
    command.add_argument("--smoke", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.dataset, args.run_id, args.smoke)


if __name__ == "__main__":
    main()
