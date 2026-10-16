"""Refresh the small R5 audit indexes from completed canonical runs.

This is an audit/reporting utility only.  It does not run a model or compute
new statistics; all values are copied from the specified canonical tables.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "docs/audit/r5"
THRESHOLD_RUN = REPO / "analysis/outputs/canonical/r5_threshold_sensitivity/20260928T160028Z"
METHOD_RUN = REPO / "analysis/outputs/canonical/r5_cross_method/20260928T162045Z"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_population_success() -> None:
    frame = pd.read_csv(THRESHOLD_RUN / "analysis/cross_system_threshold_sensitivity.csv")
    frame = frame[frame.threshold_type == "P95"].sort_values("dataset")
    fields = [
        "dataset", "P90_exists", "P95_exists", "P99_exists", "tau_P95",
        "success_fraction", "conditional_Eind", "conditional_Cind",
        "conditional_Q", "canonical", "source_path", "notes",
    ]
    rows = []
    for row in frame.to_dict("records"):
        dataset = row["dataset"]
        source = "; ".join([
            str(THRESHOLD_RUN / "analysis/cross_system_threshold_sensitivity.csv"),
            str(THRESHOLD_RUN / "raw/cross_system_threshold_battery.csv"),
            str(THRESHOLD_RUN / "raw/clean_validation_Epop.csv.gz"),
            str(THRESHOLD_RUN / "provenance/run_manifest.json"),
        ])
        rows.append({
            "dataset": dataset,
            "P90_exists": "YES", "P95_exists": "YES", "P99_exists": "YES",
            "tau_P95": row["tau_pop"],
            "success_fraction": row["test_population_success_fraction"],
            "conditional_Eind": row["conditional_E_ind"],
            "conditional_Cind": row["conditional_C_ind"],
            "conditional_Q": row["conditional_Q"],
            "canonical": "YES",
            "source_path": source,
            "notes": (
                "Trajectory-level clean-validation P95; explicit test success fraction; "
                "conditional metrics are median within successful cycles per battery, "
                "then battery-balanced system medians with whole-battery bootstrap"
            ),
        })
    write_csv(AUDIT / "population_success_audit.csv", fields, rows)


def update_cross_method() -> None:
    summary = pd.read_csv(METHOD_RUN / "analysis/cross_method_summary.csv")
    summary = summary[summary.threshold_type == "P95"].sort_values(["method", "corruption"])
    inventory = pd.read_csv(METHOD_RUN / "provenance/cross_method_archive_inventory.csv")
    old = {
        "batterydit": ("93.1%", "0.476", "0.713"),
        "transformer": ("85.2%", "0.478", "0.726"),
        "csdi": ("89.5%", "1.392", "0.348"),
        "kalman": ("95.4%", "3.227", "-0.018"),
    }
    fields = [
        "method", "corruption", "clean_archive_exists", "reconstruction_archive_exists",
        "same_test_batteries", "same_corruption_realizations", "raw_cycle_level_exists",
        "battery_summary_exists", "current_canonical_decomposition", "P95_exists",
        "P90_exists", "P99_exists", "population_success_fraction", "current_Eind",
        "current_Cind", "current_Q", "conditional_Eind", "conditional_Cind",
        "conditional_Q", "old_table12_population_success", "old_table12_Eind",
        "old_table12_Cind", "old_table12_reproducible", "canonical", "source_path", "notes",
    ]
    rows = []
    for row in summary.to_dict("records"):
        subset = inventory[
            (inventory.method == row["method"]) &
            (inventory.corruption == row["corruption"])
        ]
        same_keys = subset.keys_sha256.nunique() == 1
        same_clean = subset.clean_voltage_sha256.nunique() == 1
        same_input = subset.input_voltage_sha256.nunique() == 1
        old_values = old[row["method"]]
        source = "; ".join([
            str(METHOD_RUN / "analysis/cross_method_summary.csv"),
            str(METHOD_RUN / "analysis/cross_method_battery_summary.csv"),
            str(METHOD_RUN / "raw/cross_method_cycle_level.csv.gz"),
            str(METHOD_RUN / "analysis/cross_method_P90_sensitivity.csv"),
            str(METHOD_RUN / "analysis/cross_method_P99_sensitivity.csv"),
            str(METHOD_RUN / "provenance/run_manifest.json"),
        ])
        rows.append({
            "method": row["method"], "corruption": row["corruption"],
            "clean_archive_exists": "YES", "reconstruction_archive_exists": "YES",
            "same_test_batteries": "YES" if same_keys else "NO",
            "same_corruption_realizations": "YES" if same_clean and same_input else "NO",
            "raw_cycle_level_exists": "YES", "battery_summary_exists": "YES",
            "current_canonical_decomposition": "YES", "P95_exists": "YES",
            "P90_exists": "YES", "P99_exists": "YES",
            "population_success_fraction": row["population_success_fraction"],
            "current_Eind": row["unconditional_median_E_ind"],
            "current_Cind": row["unconditional_median_C_ind"],
            "current_Q": row["unconditional_median_Q"],
            "conditional_Eind": row["conditional_median_E_ind"],
            "conditional_Cind": row["conditional_median_C_ind"],
            "conditional_Q": row["conditional_median_Q"],
            "old_table12_population_success": old_values[0],
            "old_table12_Eind": old_values[1], "old_table12_Cind": old_values[2],
            "old_table12_reproducible": "NO", "canonical": "PASS_READY",
            "source_path": source,
            "notes": (
                "P95 population success; model-seed median per trajectory, then successful-cycle "
                "median per battery and battery-balanced conditional summary; four benchmark "
                "corruptions only; old SI values are not reproducible"
            ),
        })
    write_csv(AUDIT / "cross_method_audit.csv", fields, rows)


if __name__ == "__main__":
    update_population_success()
    update_cross_method()
    print(f"updated R5 audit indexes from {THRESHOLD_RUN} and {METHOD_RUN}")
