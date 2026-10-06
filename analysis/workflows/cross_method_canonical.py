"""R5-F canonical cross-method population/individual analysis.

The workflow consumes frozen MIT reconstruction archives for BatteryDiT,
Transformer, CSDI and Kalman.  It is restricted to Downsampling, Random,
Segment and Destructive corruption.  No Hybrid archive is read here.

For each method/corruption the aggregation is explicit and reproducible:

    model-seed trajectories -> median per cell/cycle
    -> population-success mask at P90/P95/P99
    -> median over successful cycles within each battery
    -> battery-balanced system summary and whole-battery bootstrap

This is analysis-only.  Missing archives are reported as upstream gaps rather
than silently replaced by old Supplementary Table 12 values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _certified as C
from . import _stats as S
from .cross_system import validation_metrics_and_thresholds
from .provenance import canonical_run

METHODS = {
    "batterydit": (2021, 2022, 2023),
    "transformer": (2021, 2022, 2023),
    "csdi": (2021, 2022, 2023),
    "kalman": (2021,),
}
BENCHMARK_CORRUPTIONS = ("downsample", "random", "segment", "destructive")
PERCENTILES = (90, 95, 99)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260922
METRICS = ("E_pop", "E_ind", "C_ind", "Q", "waveform_rmse")


def _sha256_bytes(value: np.ndarray | None) -> str:
    if value is None:
        return "NA"
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _archive_record(path: str, archive: dict[str, np.ndarray]) -> dict:
    return {
        "archive_path": path,
        "keys_sha256": _sha256_bytes(np.asarray(archive.get("keys"))),
        "clean_voltage_sha256": _sha256_bytes(archive.get("clean_voltage")),
        "input_voltage_sha256": _sha256_bytes(archive.get("input_voltage")),
        "n_trajectories": int(len(archive.get("keys", []))),
        "clean_shape": list(np.asarray(archive["clean_voltage"]).shape)
        if "clean_voltage" in archive else [],
        "reconstruction_shape": list(np.asarray(archive["reconstruction"]).shape)
        if "reconstruction" in archive else [],
    }


def _archive_path(dataset: str, method: str, seed: int, condition: str) -> str:
    return str(C.reconstruction_archive(dataset, condition, method, seed))


def _ordered_reconstruction(
    dataset: str, method: str, seed: int, condition: str, identity,
) -> tuple[np.ndarray, dict]:
    archive = C.load_reconstruction(dataset, condition, method, seed)
    keys = np.asarray(archive["keys"]).astype(str)
    lookup = {
        (key.rsplit("/", 1)[0], int(key.rsplit("/", 1)[1].removesuffix(".npz"))): index
        for index, key in enumerate(keys)
    }
    try:
        order = np.array([lookup[(cell, int(cycle))] for cell, cycle in identity])
    except KeyError as error:
        raise ValueError(
            f"{method} seed {seed} {condition} archive does not cover the MIT test keys"
        ) from error
    path = f"smoke://{method}/{seed}/{condition}"
    if dataset != "smoke":
        path = _archive_path(dataset, method, seed, condition)
    record = _archive_record(path, archive)
    record.update({"method": method, "model_seed": seed, "corruption": condition})
    return np.asarray(archive["reconstruction"], dtype=np.float32)[order], record


def _synthetic_clean(dataset: str, split: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    cells, cycles, voltage, soh, keys = [], [], [], [], []
    for cell in range(1, 3):
        for cycle in range(1, 11):
            cells.append(f"smoke{cell}")
            cycles.append(cycle)
            voltage.append(rng.normal(3, 0.01, (60, 1)).astype(np.float32))
            soh.append(float(1.0 - 0.01 * cycle))
            keys.append(f"smoke{cell}/{cycle}.npz")
    return {
        "voltage": np.stack(voltage), "soh": np.asarray(soh, dtype=np.float32),
        "keys": np.asarray(keys), "cells": np.asarray(cells),
        "cycles": np.asarray(cycles, dtype=np.int32),
    }


def _synthetic_reconstruction() -> dict[str, np.ndarray]:
    clean = np.random.default_rng(8).normal(3, 0.01, (20, 60, 1)).astype(np.float32)
    return {
        "keys": np.asarray([
            f"smoke{cell}/{cycle}.npz"
            for cell in range(1, 3) for cycle in range(1, 11)
        ]),
        "clean_voltage": clean,
        "input_voltage": clean.copy(),
        "reconstruction": clean + np.random.default_rng(9).normal(
            0, 0.001, clean.shape
        ).astype(np.float32),
    }


def _battery_summary(trajectory: pd.DataFrame, method: str, corruption: str,
                     threshold_type: str, tau_pop: float) -> pd.DataFrame:
    success = trajectory[trajectory["population_success"]].copy()
    all_battery = trajectory.groupby("cell", as_index=False, sort=True).agg(
        n_test_trajectories=("E_pop", "size"),
        n_success_trajectories=("population_success", "sum"),
    )
    all_battery["population_success_fraction"] = (
        all_battery["n_success_trajectories"] /
        all_battery["n_test_trajectories"]
    )
    all_battery["has_population_success"] = all_battery["n_success_trajectories"] > 0
    if success.empty:
        conditional = pd.DataFrame({"cell": all_battery["cell"]})
        for metric in METRICS:
            conditional[f"conditional_median_{metric}"] = np.nan
    else:
        conditional = success.groupby("cell", as_index=False, sort=True).agg(
            **{f"conditional_median_{metric}": (metric, "median") for metric in METRICS},
        )
    output = all_battery.merge(conditional, on="cell", how="left", sort=True)
    output.insert(0, "method", method)
    output.insert(1, "corruption", corruption)
    output.insert(2, "threshold_type", threshold_type)
    output.insert(3, "tau_pop", float(tau_pop))
    return output


def _battery_median_ci(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    values = frame[column].dropna()
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    batteries = frame.loc[values.index, "cell"]
    return S.battery_bootstrap(
        values.to_numpy(dtype=float), batteries.to_numpy(), np.median,
        n=BOOTSTRAPS, seed=BOOTSTRAP_SEED,
    )


def _summary_row(
    trajectory: pd.DataFrame, battery: pd.DataFrame, method: str,
    corruption: str, threshold_type: str, tau_pop: float, seeds: tuple[int, ...],
) -> dict:
    successful_batteries = battery[battery["has_population_success"]]
    row = {
        "method": method, "corruption": corruption,
        "threshold_type": threshold_type, "tau_pop": float(tau_pop),
        "n_model_seeds": int(len(seeds)), "model_seeds": json.dumps(list(seeds)),
        "n_test_trajectories": int(len(trajectory)),
        "n_success_trajectories": int(trajectory["population_success"].sum()),
        "population_success_fraction": float(trajectory["population_success"].mean()),
        "n_test_batteries": int(len(battery)),
        "n_successful_batteries": int(battery["has_population_success"].sum()),
        "successful_battery_fraction": float(battery["has_population_success"].mean()),
        "battery_balanced_population_success_fraction": float(
            battery["population_success_fraction"].mean()
        ),
        "unconditional_median_E_pop": float(trajectory["E_pop"].median()),
        "unconditional_median_E_ind": float(trajectory["E_ind"].median()),
        "unconditional_median_C_ind": float(trajectory["C_ind"].median()),
        "unconditional_median_Q": float(trajectory["Q"].median()),
        "conditional_n_batteries": int(len(successful_batteries)),
        "status": "READY",
    }
    for metric in ("E_ind", "C_ind", "Q"):
        observed, low, high = _battery_median_ci(
            successful_batteries, f"conditional_median_{metric}"
        )
        row[f"conditional_median_{metric}"] = observed
        row[f"conditional_{metric}"] = observed
        row[f"conditional_{metric}_ci_low"] = low
        row[f"conditional_{metric}_ci_high"] = high
    return row


def _trajectory_level(stacked: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    grouped = stacked.groupby(["cell", "cycle"], as_index=False, sort=True)
    trajectory = grouped[list(METRICS)].median()
    trajectory["key"] = trajectory.apply(
        lambda row: f"{row['cell']}/{int(row['cycle'])}.npz", axis=1
    )
    for percentile in PERCENTILES:
        trajectory[f"population_success_P{percentile}"] = (
            trajectory["E_pop"] <= thresholds[f"tau_pop_p{percentile}"]
        )
    return trajectory


def run(run_id: str | None = None, smoke: bool = False) -> Path:
    run = canonical_run("r5_cross_method", run_id)
    original_load_clean = C.module().load_clean
    original_load_reconstruction = C.load_reconstruction
    original_reconstruction_archive = C.reconstruction_archive
    if smoke:
        C.module().load_clean = _synthetic_clean
        C.load_reconstruction = lambda dataset, condition, method, seed: _synthetic_reconstruction()

    try:
        validation, thresholds = validation_metrics_and_thresholds("mit")
        certified = C.module()
        train = certified.load_clean("mit", "train")
        validation_block = certified.load_clean("mit", "vali")
        test = certified.load_clean("mit", "test")
        spline, model, s_ind, selection, manifold = certified.select_manifold(
            train, validation_block
        )
        state = certified.prepare_test_decomposition(test, spline, model)
        identity = list(zip(
            np.asarray(test["cells"]).astype(str),
            np.asarray(test["cycles"], dtype=int),
        ))

        raw_rows, trajectory_rows, battery_rows, summary_rows, archive_rows = [], [], [], [], []
        for method, seeds in METHODS.items():
            for corruption in BENCHMARK_CORRUPTIONS:
                frames, records = [], []
                try:
                    for seed in seeds:
                        reconstruction, record = _ordered_reconstruction(
                            "mit" if not smoke else "smoke", method, seed,
                            corruption, identity,
                        )
                        metrics, _ = certified.evaluate_reconstruction(
                            reconstruction, test, state, spline, model, s_ind,
                            train["soh"],
                        )
                        metrics["method"] = method
                        metrics["corruption"] = corruption
                        metrics["model_seed"] = seed
                        frames.append(metrics)
                        records.append(record)
                except (FileNotFoundError, KeyError, ValueError) as error:
                    for percentile in PERCENTILES:
                        summary_rows.append({
                            "method": method, "corruption": corruption,
                            "threshold_type": f"P{percentile}",
                            "status": "MISSING_UPSTREAM",
                            "notes": str(error),
                        })
                    continue

                signatures = {
                    tuple(record[key] for key in (
                        "keys_sha256", "clean_voltage_sha256", "input_voltage_sha256",
                    )) for record in records
                }
                if len(signatures) != 1:
                    raise ValueError(
                        f"archive identity mismatch for {method}/{corruption}: {records}"
                    )
                archive_rows.extend(records)
                stacked = pd.concat(frames, ignore_index=True)
                for percentile in PERCENTILES:
                    threshold_type = f"P{percentile}"
                    trajectory = _trajectory_level(stacked, thresholds)
                    trajectory["population_success"] = trajectory[
                        f"population_success_{threshold_type}"
                    ]
                    battery = _battery_summary(
                        trajectory, method, corruption, threshold_type,
                        thresholds[f"tau_pop_p{percentile}"],
                    )
                    summary_rows.append(_summary_row(
                        trajectory, battery, method, corruption, threshold_type,
                        thresholds[f"tau_pop_p{percentile}"], seeds,
                    ))
                    trajectory_copy = trajectory.copy()
                    trajectory_copy.insert(0, "method", method)
                    trajectory_copy.insert(1, "corruption", corruption)
                    trajectory_copy.insert(2, "threshold_type", threshold_type)
                    trajectory_copy["tau_pop"] = thresholds[
                        f"tau_pop_p{percentile}"
                    ]
                    trajectory_rows.append(trajectory_copy)
                    battery_rows.append(battery)
                for frame in frames:
                    frame = frame.copy()
                    frame["population_success_P90"] = (
                        frame["E_pop"] <= thresholds["tau_pop_p90"]
                    )
                    frame["population_success_P95"] = (
                        frame["E_pop"] <= thresholds["tau_pop_p95"]
                    )
                    frame["population_success_P99"] = (
                        frame["E_pop"] <= thresholds["tau_pop_p99"]
                    )
                    raw_rows.append(frame)

        raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
        trajectory = pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame()
        battery = pd.concat(battery_rows, ignore_index=True) if battery_rows else pd.DataFrame()
        summary = pd.DataFrame(summary_rows)
        archive_inventory = pd.DataFrame(archive_rows)
        validation_out = validation[["cell", "cycle", "E_pop"]].copy()
        validation_out.insert(0, "dataset", "MIT")

        run.write_table("raw", "clean_validation_Epop.csv", validation_out)
        run.write_table("raw", "cross_method_cycle_level.csv.gz", raw)
        run.write_table("raw", "cross_method_trajectory_level.csv.gz", trajectory)
        run.write_table("analysis", "cross_method_battery_summary.csv", battery)
        run.write_table("analysis", "cross_method_summary.csv", summary)
        for percentile in PERCENTILES:
            run.write_table(
                "analysis", f"cross_method_P{percentile}_sensitivity.csv",
                summary[summary["threshold_type"] == f"P{percentile}"],
            )
        run.write_table(
            "plotting", "cross_method_plotting.csv",
            summary[summary["threshold_type"] == "P95"][[
                "method", "corruption", "population_success_fraction",
                "battery_balanced_population_success_fraction",
                "conditional_median_E_ind", "conditional_median_C_ind",
                "conditional_median_Q",
            ]],
        )
        run.write_table("provenance", "cross_method_archive_inventory.csv", archive_inventory)
        run.write_table(
            "provenance", "cross_method_thresholds.csv",
            pd.DataFrame([
                {
                    "dataset": "MIT", "threshold_type": f"P{percentile}",
                    "tau_pop": thresholds[f"tau_pop_p{percentile}"],
                    "validation_n_batteries": thresholds["n_validation_batteries"],
                    "validation_n_trajectories": thresholds["n_validation_trajectories"],
                    "threshold_definition": thresholds["threshold_definition"],
                }
                for percentile in PERCENTILES
            ]),
        )

        config = {
            "dataset": "MIT",
            "corruptions": list(BENCHMARK_CORRUPTIONS),
            "hybrid_included": False,
            "methods": {method: list(seeds) for method, seeds in METHODS.items()},
            "thresholds": {f"P{p}": thresholds[f"tau_pop_p{p}"] for p in PERCENTILES},
            "threshold_rule": thresholds["threshold_definition"],
            "seed_aggregation": "median per cell/cycle across available model seeds",
            "conditional_aggregation": (
                "successful cycles -> per-battery median -> battery-balanced median"
            ),
            "bootstrap": {"unit": "battery", "n": BOOTSTRAPS, "seed": BOOTSTRAP_SEED},
            "decomposition": "source_code/exp/population_individual.py",
        }
        run.write_json("cross_method_config.json", config)
        input_paths = {}
        checkpoint_ids = {}
        for record in archive_rows:
            key = f"{record['method']}_{record['corruption']}_{record['model_seed']}"
            input_paths[key] = record["archive_path"]
            checkpoint_ids[key] = record["clean_voltage_sha256"]
        run.write_manifest(
            config=config,
            input_paths=input_paths,
            checkpoint_ids=checkpoint_ids,
            seeds=sorted({seed for values in METHODS.values() for seed in values}),
            split_manifest=json.dumps(C.split_manifest("smoke" if smoke else "mit"), sort_keys=True),
        )
        print(f"R5 cross-method outputs -> {run.root}")
        return run.root
    finally:
        C.module().load_clean = original_load_clean
        C.load_reconstruction = original_load_reconstruction
        C.reconstruction_archive = original_reconstruction_archive


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
