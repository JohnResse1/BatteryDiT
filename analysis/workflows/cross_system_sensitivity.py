"""Canonical R5-D/G population-success threshold sensitivity.

This workflow reads the six frozen PopulationIndividual decompositions and
writes a complete four-layer record.  It never trains, reconstructs, refits a
manifold for test data, or bootstraps at cycle level.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import _certified as C
from .cross_system import (
    _conditional_metrics,
    _load_cycles,
    validation_metrics_and_thresholds,
)
from .provenance import canonical_run


PERCENTILES = (90, 95, 99)


def _synthetic_threshold(dataset):
    return pd.DataFrame({
        "cell": ["smoke1", "smoke2"],
        "cycle": [6, 6],
        "E_pop": [0.1, 0.2],
    }), {
        "dataset": dataset, "tau_pop_p90": 0.1, "tau_pop_p95": 0.2,
        "tau_pop_p99": 0.3, "n_validation_batteries": 2,
        "n_validation_trajectories": 2,
        "threshold_definition": "Pxx of clean-validation trajectory-level E_pop",
        "manifold_status": "SMOKE",
    }


def _synthetic_cycles() -> pd.DataFrame:
    rows = []
    for cell in ("smoke1", "smoke2"):
        for cycle in range(1, 11):
            rows.append({
                "cell": cell, "cycle": cycle, "condition": "downsample",
                "severity": "L1", "E_pop": 0.1 + cycle / 1000,
                "E_ind": 0.4, "C_ind": 0.7, "Q": 0.98,
                "waveform_rmse": 0.01, "soh_error": 0.02,
                "signature_error": 0.03,
            })
    return pd.DataFrame(rows)


def run(
    run_id: str | None = None,
    datasets: list[str] | None = None,
    smoke: bool = False,
) -> Path:
    run = canonical_run("r5_threshold_sensitivity", run_id)
    systems = ["smoke"] if smoke else (datasets or list(C.SYSTEMS))
    original_systems = C.SYSTEMS
    if smoke:
        C.SYSTEMS = {**C.SYSTEMS, "smoke": ("SMOKE", 1)}
    threshold_rows, overall_rows, metric_rows, battery_rows = [], [], [], []
    validation_rows, input_inventory = [], []

    for dataset in systems:
        if smoke:
            validation, thresholds = _synthetic_threshold(dataset)
            cycles = _synthetic_cycles()
            label = "SMOKE"
            split_manifest = {"train": {}, "vali": {}, "test": {}}
            input_path = "/tmp/smoke_population/cycle_level_decomposition.csv.gz"
        else:
            label = C.SYSTEMS[dataset][0]
            validation, thresholds = validation_metrics_and_thresholds(dataset)
            cycles = _load_cycles(dataset)
            directory = C.population_individual_run(dataset, C.SYSTEMS[dataset][1])
            input_path = str(directory / "cycle_level_decomposition.csv.gz")
            split_manifest = C.split_manifest(dataset)

        validation_rows.append(validation[["cell", "cycle", "E_pop"]].assign(dataset=label))
        input_inventory.append({
            "dataset": label, "dataset_key": dataset,
            "cycle_level_source": input_path,
            "n_cycle_rows": int(len(cycles)),
            "n_test_batteries": int(cycles["cell"].nunique()),
            "split_manifest": json.dumps(split_manifest, sort_keys=True),
        })

        for percentile in PERCENTILES:
            threshold_type = f"P{percentile}"
            threshold = thresholds[f"tau_pop_p{percentile}"]
            battery, metric, overall = _conditional_metrics(
                dataset, cycles, threshold, threshold_type
            )
            threshold_record = {
                "dataset": label, "dataset_key": dataset,
                "threshold_type": threshold_type, "tau_pop": threshold,
                "validation_n_batteries": thresholds["n_validation_batteries"],
                "validation_n_trajectories": thresholds["n_validation_trajectories"],
                "threshold_definition": thresholds["threshold_definition"],
                "manifold_status": thresholds["manifold_status"],
            }
            threshold_rows.append(threshold_record)
            overall_rows.append(overall)
            metric_rows.append(metric)
            battery_rows.append(battery)

    threshold_frame = pd.DataFrame(threshold_rows)
    overall_frame = pd.DataFrame(overall_rows)
    metrics_frame = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    battery_frame = pd.concat(battery_rows, ignore_index=True) if battery_rows else pd.DataFrame()

    if validation_rows:
        run.write_table(
            "raw", "clean_validation_Epop.csv.gz",
            pd.concat(validation_rows, ignore_index=True),
        )
    if battery_frame.shape[0]:
        run.write_table("raw", "cross_system_threshold_battery.csv", battery_frame)
    run.write_table("raw", "cross_system_threshold_inputs.csv", pd.DataFrame(input_inventory))
    run.write_table("provenance", "cross_system_thresholds.csv", threshold_frame)
    run.write_table("analysis", "cross_system_threshold_sensitivity.csv", overall_frame)
    run.write_table("analysis", "cross_system_threshold_metric_summary.csv", metrics_frame)
    run.write_table(
        "plotting", "cross_system_threshold_sensitivity.csv",
        overall_frame[[
            "dataset", "threshold_type", "tau_pop",
            "test_population_success_fraction",
            "battery_balanced_population_success_fraction",
            "conditional_E_ind", "conditional_C_ind", "conditional_Q",
        ]],
    )

    config = {
        "datasets": list(C.SYSTEMS) if not smoke else ["smoke"],
        "excluded_dataset": {"sdu": "second-life capacity target is not nominal-capacity SOH"},
        "threshold_percentiles": list(PERCENTILES),
        "threshold_definition": "Pxx of clean-validation trajectory-level E_pop",
        "population_success": "E_pop <= tau_pop(dataset)",
        "conditional_aggregation": (
            "cycle success -> median within successful cycles per battery -> "
            "battery-balanced system median; whole-battery bootstrap"
        ),
        "bootstrap": {"unit": "battery", "n": 2000, "seed": 20260921},
    }
    run.write_json("cross_system_threshold_config.json", config)
    run.write_manifest(
        config=config,
        input_paths={row["dataset_key"]: row["cycle_level_source"] for row in input_inventory},
        seeds=[C.SYSTEMS[d][1] for d in systems if d in C.SYSTEMS],
        split_manifest=json.dumps(
            {row["dataset_key"]: json.loads(row["split_manifest"]) for row in input_inventory},
            sort_keys=True,
        ),
    )
    C.SYSTEMS = original_systems
    print(f"R5 threshold sensitivity outputs -> {run.root}")
    return run.root


def parser():
    import argparse

    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    command.add_argument("--datasets", nargs="*", default=None)
    command.add_argument("--smoke", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.run_id, args.datasets, args.smoke)


if __name__ == "__main__":
    main()
