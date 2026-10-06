"""R4-01 canonical longitudinal continuity, restricted to the frozen claim.

Outputs only:
1. same-battery individual-state distance vs lag;
2. SOH-matched cross-battery control.

No fingerprint, identity, ageing-pace, velocity/acceleration or hidden-state
claim is computed. PCA, when used, is fit on train batteries only and applied
frozen to validation/test.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import _certified as C
from .provenance import canonical_run


def _fit_train_pca(train_sequences) -> dict:
    matrix = np.concatenate([seq["residual"] for seq in train_sequences], axis=0)
    mean = matrix.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(matrix - mean, full_matrices=False)
    return {"mean": mean, "components": components[:C.PRIMARY_K]}


def _project(pca: dict, residual: np.ndarray) -> np.ndarray:
    return (residual - pca["mean"]) @ pca["components"].T


def _synthetic_sequences():
    rng = np.random.default_rng(7)
    def make(split):
        seqs=[]
        for cell_index in range(3):
            residual=rng.normal(0,1,(20,8))
            soh=np.linspace(0.98,0.90,20)
            cycles=np.arange(1,21)
            seqs.append({"cell":f"smoke{cell_index+1}","batch":"sm","cycle":cycles,"soh":soh,"residual":residual})
        return seqs
    return {"train":make("train"),"test":make("test")}


def run(dataset: str = "mit", run_id: str | None = None, lags=(1, 5, 10, 25, 50, 100), smoke: bool = False) -> Path:
    run = canonical_run("r4_longitudinal_continuity", run_id)
    if smoke:
        C.load_sequences = lambda dataset, splits: _synthetic_sequences()
    sequences = C.load_sequences(dataset, ("train", "test"))
    pca = _fit_train_pca(sequences["train"])

    raw_rows = []
    for sequence in sequences["test"]:
        z = _project(pca, sequence["residual"])
        cells = np.repeat(sequence["cell"], len(z))
        cycles = np.asarray(sequence["cycle"], dtype=int)
        soh = np.asarray(sequence["soh"], dtype=float)
        for lag in lags:
            if len(z) <= lag:
                continue
            distance = np.linalg.norm(z[:-lag] - z[lag:], axis=1)
            for i in range(len(distance)):
                raw_rows.append({
                    "battery_id": sequence["cell"],
                    "cycle_i": int(cycles[i]),
                    "cycle_j": int(cycles[i + lag]),
                    "lag": int(lag),
                    "soh_i": float(soh[i]),
                    "soh_j": float(soh[i + lag]),
                    "same_battery": True,
                    "distance": float(distance[i]),
                })

    # SOH-matched cross-battery control uses the pre-registered anchor rule.
    rng = np.random.default_rng(C.SEED)
    test_sequences = sequences["test"]
    for lag in lags:
        for i, sequence in enumerate(test_sequences):
            z_i = _project(pca, sequence["residual"])
            if len(z_i) <= lag:
                continue
            indices = rng.choice(len(z_i), size=min(40, len(z_i)), replace=False)
            for index in indices:
                soh_i = sequence["soh"][index]
                hits: list[tuple[str, int, float, float, float]] = []
                for other in test_sequences:
                    if other["cell"] == sequence["cell"]:
                        continue
                    z_other = _project(pca, other["residual"])
                    mask = np.abs(other["soh"] - soh_i) <= C.ANCHOR_HALF_WIDTH
                    if not mask.any():
                        continue
                    other_index = int(np.flatnonzero(mask)[0])
                    distance = float(np.linalg.norm(z_i[index] - z_other[other_index]))
                    hits.append((other["cell"], int(other["cycle"][other_index]),
                                 float(other["soh"][other_index]), distance, distance))
                if hits:
                    cell, cycle, soh, distance, _ = hits[0]
                    raw_rows.append({
                        "battery_id": sequence["cell"],
                        "cycle_i": int(sequence["cycle"][index]),
                        "cycle_j": int(cycle),
                        "lag": int(lag),
                        "soh_i": float(soh_i),
                        "soh_j": float(soh),
                        "same_battery": False,
                        "distance": distance,
                    })

    raw = pd.DataFrame(raw_rows)
    run.write_table("raw", "longitudinal_continuity_raw.csv.gz", raw)
    analysis = raw.groupby(["lag", "same_battery"], as_index=False).agg(
        distance_median=("distance", "median"),
        distance_mean=("distance", "mean"),
        distance_q25=("distance", lambda x: float(np.percentile(x, 25))),
        distance_q75=("distance", lambda x: float(np.percentile(x, 75))),
        n_observations=("distance", "size"))
    pivot = analysis.pivot(index="lag", columns="same_battery", values="distance_median")
    analysis["cross_over_same_ratio"] = analysis.apply(
        lambda row: float(pivot.loc[row.lag, False] / pivot.loc[row.lag, True])
        if row.same_battery and True in pivot.columns and False in pivot.columns else np.nan, axis=1)
    run.write_table("analysis", "longitudinal_continuity_summary.csv", analysis)
    run.write_table("plotting", "longitudinal_continuity_plotting.csv", analysis[
        analysis.same_battery == True][["lag", "distance_median"]].rename(
        columns={"distance_median": "same_battery_distance"}).merge(
        analysis[analysis.same_battery == False][["lag", "distance_median"]].rename(
            columns={"distance_median": "cross_battery_distance"}), on="lag", how="left"))
    run.write_manifest(
        config={"dataset": dataset, "lags": list(lags), "primary_k": C.PRIMARY_K,
                "anchor_half_width": C.ANCHOR_HALF_WIDTH},
        input_paths={"clean_data": f"source_code/data/{dataset}/{{train,test}}/clean"},
        seeds=[C.SYSTEMS[dataset][1]])
    print(f"R4 longitudinal outputs -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset", default="mit")
    command.add_argument("--run-id")
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.dataset, args.run_id)


if __name__ == "__main__":
    main()
