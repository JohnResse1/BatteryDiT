"""Canonical cross-system population/individual recoverability analysis.

R5-1  six-system certified population-individual decomposition
R5-2  cross-system recoverability asymmetry
R5-3  individual fidelity conditional on population success
R5-4  individual loss versus degradation-relevant downstream error

The primary population-success rule is frozen as::

    tau_pop(dataset) = P95(clean-validation trajectory-level E_pop)
    population_success = E_pop <= tau_pop(dataset)

All conditional summaries are battery-balanced: cycle-level success is
computed first, then successful cycles are reduced to one median per battery,
and only then is the system summary/whole-battery bootstrap calculated.
SDU is intentionally absent because its target is dataset-defined
second-life capacity rather than compatible nominal-capacity SOH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.utils.paths import ANALYSIS_OUTPUT_ROOT
from . import _certified as C
from . import _stats as S

THRESHOLD_PERCENTILE = 95
FORMAL_E_IND_HIGH = 1.0
FORMAL_Q_LOW, FORMAL_Q_HIGH = 0.95, 1.05
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260921
CYCLE_METRICS = (
    "E_pop", "E_ind", "C_ind", "Q", "waveform_rmse", "soh_error",
    "signature_error",
)


def _label(dataset: str) -> str:
    return C.SYSTEMS[dataset][0]


def _load_cycles(dataset: str) -> pd.DataFrame:
    directory = C.population_individual_run(dataset, C.SYSTEMS[dataset][1])
    path = directory / "cycle_level_decomposition.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"cell", "cycle", "condition", "severity", *CYCLE_METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing certified decomposition columns {sorted(missing)}")
    return frame


def validation_metrics_and_thresholds(dataset: str) -> tuple[pd.DataFrame, dict]:
    """Return clean-validation ``E_pop`` trajectories and P90/P95/P99."""
    metrics, state = C.clean_validation_metrics(dataset)
    values = metrics["E_pop"].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"clean validation E_pop is empty/non-finite for {dataset}")
    thresholds = {
        "dataset": dataset,
        "tau_pop_p90": float(np.percentile(values, 90)),
        "tau_pop_p95": float(np.percentile(values, 95)),
        "tau_pop_p99": float(np.percentile(values, 99)),
        "n_validation_batteries": int(metrics["cell"].nunique()),
        "n_validation_trajectories": int(len(metrics)),
        "threshold_definition": "Pxx of clean-validation trajectory-level E_pop",
        "manifold_status": str(state["manifold"].get("manifold_status", "UNKNOWN")),
    }
    return metrics, thresholds


def validation_thresholds(dataset: str) -> dict:
    """Compute P90/P95/P99 from clean validation trajectory-level ``E_pop``."""
    _, thresholds = validation_metrics_and_thresholds(dataset)
    return thresholds


def _bootstrap_median(values: pd.Series, batteries: pd.Series) -> tuple[float, float, float]:
    finite = np.isfinite(values.to_numpy(dtype=float))
    values = values.iloc[np.flatnonzero(finite)]
    batteries = batteries.iloc[np.flatnonzero(finite)]
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    return S.battery_bootstrap(
        values.to_numpy(dtype=float), batteries.to_numpy(), np.median,
        n=BOOTSTRAPS, seed=BOOTSTRAP_SEED,
    )


def _conditional_metrics(
    dataset: str, cycles: pd.DataFrame, threshold: float, threshold_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return battery table, metric table, and one explicit overall row."""
    frame = cycles.copy()
    frame["population_success"] = frame["E_pop"] <= float(threshold)
    frame["E_ind_high"] = frame["E_ind"] >= FORMAL_E_IND_HIGH
    frame["Q_outside_band"] = (frame["Q"] < FORMAL_Q_LOW) | (frame["Q"] > FORMAL_Q_HIGH)
    frame["C_ind_nonpositive"] = frame["C_ind"] <= 0
    success = frame[frame["population_success"]].copy()

    all_battery = frame.groupby("cell", as_index=False, sort=True).agg(
        n_test_trajectories=("E_pop", "size"),
        n_success_trajectories=("population_success", "sum"),
    )
    all_battery["population_success_fraction"] = (
        all_battery["n_success_trajectories"] /
        all_battery["n_test_trajectories"]
    )
    all_battery["has_population_success"] = all_battery["n_success_trajectories"] > 0

    if success.empty:
        success_battery = pd.DataFrame({"cell": all_battery["cell"]})
        for metric in CYCLE_METRICS:
            success_battery[f"median_{metric}"] = np.nan
        for flag in ("E_ind_high", "Q_outside_band", "C_ind_nonpositive"):
            success_battery[f"fraction_{flag}"] = np.nan
    else:
        success_battery = success.groupby("cell", as_index=False, sort=True).agg(
            **{f"median_{metric}": (metric, "median") for metric in CYCLE_METRICS},
            **{
                f"fraction_{flag}": (flag, "mean")
                for flag in ("E_ind_high", "Q_outside_band", "C_ind_nonpositive")
            },
        )
    battery = all_battery.merge(success_battery, on="cell", how="left", sort=True)
    battery.insert(0, "dataset", _label(dataset))
    battery.insert(1, "threshold_type", threshold_type)
    battery.insert(2, "tau_pop", float(threshold))

    metric_rows = []
    successful_batteries = battery[battery["has_population_success"]].copy()
    for metric in CYCLE_METRICS:
        column = f"median_{metric}"
        values = successful_batteries[column].dropna()
        batteries = successful_batteries.loc[values.index, "cell"]
        observed, low, high = _bootstrap_median(values, batteries)
        metric_rows.append({
            "dataset": _label(dataset), "threshold_type": threshold_type,
            "tau_pop": float(threshold), "metric": metric,
            "median": observed, "median_ci_low": low,
            "median_ci_high": high,
            "n_batteries": int(len(successful_batteries)),
        })
    for flag in ("E_ind_high", "Q_outside_band", "C_ind_nonpositive"):
        column = f"fraction_{flag}"
        values = successful_batteries[column].dropna()
        batteries = successful_batteries.loc[values.index, "cell"]
        observed, low, high = _bootstrap_median(values, batteries)
        metric_rows.append({
            "dataset": _label(dataset), "threshold_type": threshold_type,
            "tau_pop": float(threshold), "metric": column,
            "median": observed, "median_ci_low": low,
            "median_ci_high": high,
            "n_batteries": int(len(successful_batteries)),
        })

    overall = {
        "dataset": _label(dataset), "dataset_key": dataset,
        "threshold_type": threshold_type, "tau_pop": float(threshold),
        "n_test_trajectories": int(len(frame)),
        "n_success_trajectories": int(len(success)),
        "test_population_success_fraction": float(frame["population_success"].mean()),
        "n_test_batteries": int(len(battery)),
        "n_successful_batteries": int(battery["has_population_success"].sum()),
        "successful_battery_fraction": float(battery["has_population_success"].mean()),
        "battery_balanced_population_success_fraction": float(
            battery["population_success_fraction"].mean()
        ),
    }
    by_metric = {row["metric"]: row for row in metric_rows}
    for metric in ("E_ind", "C_ind", "Q"):
        row = by_metric[metric]
        overall[f"conditional_{metric}"] = row["median"]
        overall[f"conditional_{metric}_ci_low"] = row["median_ci_low"]
        overall[f"conditional_{metric}_ci_high"] = row["median_ci_high"]
    return battery, pd.DataFrame(metric_rows), overall


def conditional_summary(dataset: str, threshold: float, threshold_type: str = "P95") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Battery-balanced conditional metrics, preserving the legacy API."""
    battery, summary, _ = _conditional_metrics(
        dataset, _load_cycles(dataset), threshold, threshold_type
    )
    return battery, summary


def condition_rollup(dataset: str, threshold: float) -> pd.DataFrame:
    """Per-system/corruption/severity conditional individual summaries."""
    cycles = _load_cycles(dataset)
    cycles["population_success"] = cycles["E_pop"] <= float(threshold)
    rows = []
    for (condition, severity), group in cycles.groupby(["condition", "severity"], sort=True):
        success = group[group["population_success"]]
        if success.empty:
            continue
        rows.append({
            "dataset": _label(dataset), "condition": condition, "severity": severity,
            "n_cycles": int(len(success)),
            "conditional_median_E_ind": float(success["E_ind"].median()),
            "conditional_median_C_ind": float(success["C_ind"].median()),
            "conditional_median_soh_error": float(success["soh_error"].median()),
            "conditional_median_signature_error": float(success["signature_error"].median()),
            "fraction_E_ind_high": float((success["E_ind"] >= FORMAL_E_IND_HIGH).mean()),
            "fraction_C_ind_nonpositive": float((success["C_ind"] <= 0).mean()),
        })
    return pd.DataFrame(rows)


def downstream_linkage(dataset: str, threshold: float) -> pd.DataFrame:
    """Battery-balanced conditional E_ind/linkage inventory for R5-4."""
    cycles = _load_cycles(dataset)
    success = cycles[cycles["E_pop"] <= float(threshold)]
    if success.empty:
        return pd.DataFrame()
    per_battery = success.groupby("cell", as_index=False).agg(
        E_ind=("E_ind", "median"), C_ind=("C_ind", "median"),
        soh_error=("soh_error", "median"), signature_error=("signature_error", "median"),
    )
    rows = []
    for x_column, y_column in (
        ("E_ind", "soh_error"), ("E_ind", "signature_error"),
        ("C_ind", "signature_error"), ("C_ind", "soh_error"),
    ):
        rho, low, high = S.battery_bootstrap_pair(
            per_battery[x_column].to_numpy(dtype=float),
            per_battery[y_column].to_numpy(dtype=float),
            per_battery["cell"].to_numpy(), n=BOOTSTRAPS, seed=BOOTSTRAP_SEED,
        )
        rows.append({
            "dataset": _label(dataset), "x": x_column, "y": y_column,
            "spearman_rho": rho, "bootstrap_low": low, "bootstrap_high": high,
            "n_batteries": int(len(per_battery)),
        })
    return pd.DataFrame(rows)


def run() -> None:
    C.log("=" * 78); C.log("Cross-system conditional asymmetry"); C.log("=" * 78)
    threshold_rows, summary_rows, battery_rows = [], [], []
    rollup_rows, linkage_rows, overall_rows = [], [], []
    for dataset, (label, seed) in C.SYSTEMS.items():
        try:
            thresholds = validation_thresholds(dataset)
            cycles = _load_cycles(dataset)
            battery, summary, overall = _conditional_metrics(
                dataset, cycles, thresholds["tau_pop_p95"], "P95"
            )
        except (FileNotFoundError, ValueError) as error:
            C.log(f"  {label}: unavailable ({error}); skipped")
            continue
        threshold_rows.append(thresholds)
        summary_rows.append(summary)
        battery_rows.append(battery)
        overall_rows.append(overall)
        rollup_rows.append(condition_rollup(dataset, thresholds["tau_pop_p95"]))
        linkage_rows.append(downstream_linkage(dataset, thresholds["tau_pop_p95"]))
        C.log(
            f"  {label}: tau_pop={thresholds['tau_pop_p95']:.5f} "
            f"| test success={overall['test_population_success_fraction']:.3f} "
            f"| conditional E_ind={overall['conditional_E_ind']:.3f} "
            f"C_ind={overall['conditional_C_ind']:.3f} "
            f"| {overall['n_test_batteries']} batteries"
        )

    pd.DataFrame(threshold_rows).to_csv(
        ANALYSIS_OUTPUT_ROOT / "cross_system_thresholds.csv", index=False
    )
    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(
            ANALYSIS_OUTPUT_ROOT / "cross_system_conditional_summary.csv", index=False
        )
    if battery_rows:
        pd.concat(battery_rows, ignore_index=True).to_csv(
            ANALYSIS_OUTPUT_ROOT / "cross_system_conditional_battery.csv", index=False
        )
    if overall_rows:
        pd.DataFrame(overall_rows).to_csv(
            ANALYSIS_OUTPUT_ROOT / "cross_system_conditional_overall.csv", index=False
        )
    if rollup_rows:
        rollup = pd.concat(rollup_rows, ignore_index=True)
        rollup.to_csv(ANALYSIS_OUTPUT_ROOT / "cross_system_condition_rollup.csv", index=False)
        compact = rollup.groupby("dataset", as_index=False).agg(
            conditions_evaluated=("condition", "size"),
            median_condition_E_ind=("conditional_median_E_ind", "median"),
            median_condition_C_ind=("conditional_median_C_ind", "median"),
            fraction_conditions_E_ind_high=("fraction_E_ind_high", "mean"),
            fraction_conditions_C_ind_nonpositive=("fraction_C_ind_nonpositive", "mean"),
        )
        compact.to_csv(ANALYSIS_OUTPUT_ROOT / "cross_system_asymmetry_rollup.csv", index=False)
    if linkage_rows:
        nonempty = [row for row in linkage_rows if len(row)]
        if nonempty:
            pd.concat(nonempty, ignore_index=True).to_csv(
                ANALYSIS_OUTPUT_ROOT / "cross_system_downstream_linkage.csv", index=False
            )
    C.log(f"outputs -> {ANALYSIS_OUTPUT_ROOT}")


if __name__ == "__main__":
    run()
