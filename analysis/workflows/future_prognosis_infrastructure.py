"""R4-06..R4-09 infrastructure, blocked pending human target confirmation.

This module exports battery/cycle-level prediction rows and exposes a strict
predictor interface. It deliberately does not execute a full formal prognosis
run until docs/audit/r3_r5/phase2/prognosis_target_spec.md is confirmed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import _certified as C
from .provenance import canonical_run

FEATURE_SETS = ("P", "P+I", "P+H", "P+H+I")
PREDICTORS = ("ridge", "knn", "mlp")


def load_predictors():
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    return {"ridge": Ridge, "knn": KNeighborsRegressor}


def _fit_train_pca(train_sequences) -> dict:
    matrix = np.concatenate([seq["residual"] for seq in train_sequences], axis=0)
    mean = matrix.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(matrix - mean, full_matrices=False)
    return {"mean": mean, "components": components[:C.PRIMARY_K]}


def _project(pca: dict, residual: np.ndarray) -> np.ndarray:
    return (residual - pca["mean"]) @ pca["components"].T


def _slope(values: np.ndarray, window: int) -> float:
    if len(values) < 2:
        return np.nan
    values = np.asarray(values[-min(window, len(values)):], dtype=float)
    t = np.arange(len(values), dtype=float)
    denominator = np.sum((t - t.mean()) ** 2)
    if denominator <= 0:
        return np.nan
    return float(np.sum((t - t.mean()) * (values - values.mean())) / denominator)


def _synthetic_sequences():
    rng = np.random.default_rng(7)
    def make(split):
        seqs=[]
        for cell_index in range(3):
            seqs.append({"cell":f"smoke{cell_index+1}","batch":"sm","cycle":np.arange(1,21),"soh":np.linspace(0.98,0.90,20),"residual":rng.normal(0,1,(20,8))})
        return seqs
    return {"train":make("train"),"vali":make("vali"),"test":make("test")}


def build_feature_table(dataset: str = "mit", future_horizon: int = 100, smoke: bool = False) -> tuple[pd.DataFrame, dict]:
    """Construct features and future target without choosing a new target.

    The target is the historical `dSOH_<horizon>` quantity only, exported for
    schema review. It is not used for a formal run in this phase.
    """
    if smoke:
        C.load_sequences = lambda dataset, splits: _synthetic_sequences()
    sequences = C.load_sequences(dataset, ("train", "vali", "test"))
    pca = _fit_train_pca(sequences["train"])
    rows = []
    for split in ("train", "vali", "test"):
        for sequence in sequences[split]:
            z = _project(pca, sequence["residual"])
            soh = np.asarray(sequence["soh"], dtype=float)
            cycle = np.asarray(sequence["cycle"], dtype=int)
            for index in range(len(z)):
                j = np.searchsorted(cycle, cycle[index] + future_horizon)
                target = soh[j] - soh[index] if j < len(cycle) and cycle[j] == cycle[index] + future_horizon else np.nan
                rows.append({
                    "battery_id": sequence["cell"],
                    "batch": sequence["cell"][:2],
                    "split": split,
                    "cycle_id": int(cycle[index]),
                    "soh": float(soh[index]),
                    "future_target": float(target),
                    **{f"PC{component}": float(z[index, component])
                       for component in range(z.shape[1])},
                    "soh_slope_50": _slope(soh[:index + 1], 50),
                    "soh_slope_100": _slope(soh[:index + 1], 100),
                })
    frame = pd.DataFrame(rows)
    frame["P"] = frame[["soh"]].to_numpy().tolist()
    frame["I"] = frame[[f"PC{component}" for component in range(pca["components"].shape[0])]].to_numpy().tolist()
    frame["H"] = frame[["soh_slope_50", "soh_slope_100"]].to_numpy().tolist()
    frame["P+I"] = (frame["P"] + frame["I"]).to_numpy().tolist()
    frame["P+H"] = (frame["P"] + frame["H"]).to_numpy().tolist()
    frame["P+H+I"] = (frame["P"] + frame["H"] + frame["I"]).to_numpy().tolist()
    return frame, pca


def smoke(dataset: str = "mit", run_id: str | None = None, smoke: bool = True) -> Path:
    """Validate schema and predictor interface only; no formal prediction run."""
    frame, pca = build_feature_table(dataset, smoke=smoke)
    run = canonical_run("r4_future_prognosis_infrastructure", run_id)
    run.write_table("raw", "future_prognosis_features.csv.gz", frame.head(100))
    schema = {
        "raw_columns": list(frame.columns),
        "feature_sets": list(FEATURE_SETS),
        "predictors": list(PREDICTORS),
        "future_target": "dSOH_100 historical reference only",
        "status": "BLOCKED_FOR_SCIENTIFIC_CONFIRMATION",
    }
    run.write_json("future_prognosis_schema.json", schema)
    run.write_manifest(config=schema, input_paths={"clean_data": f"source_code/data/{dataset}/{{train,vali,test}}/clean"},
                       seeds=[C.SYSTEMS[dataset][1]])
    print(f"future prognosis infrastructure smoke -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset", default="mit")
    command.add_argument("--run-id")
    return command


def main() -> None:
    args = parser().parse_args()
    smoke(args.dataset, args.run_id)


if __name__ == "__main__":
    main()
