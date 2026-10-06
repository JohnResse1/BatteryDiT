"""Arrays behind the supplementary benchmark and fidelity figures.

Writes only the values the supplementary figures plot:

    supplementary_representatives.npz   one representative cycle per model x condition
    supplementary_waveform_rmse.npz     per-cycle waveform RMSE per model x condition
    supplementary_waveform_rmse_summary.npz
                                        plotting summaries for the RMSE violins
    supplementary_soh_parity.npz        predicted vs true SOH for BatteryDiT->TRB
    supplementary_soh_parity_summary.npz
                                        plotting limits for the SOH parity panels
    supplementary_downstream.npz        per-model condition means and battery errors
    supplementary_downstream_summary.npz
                                        plotting limits and grouped error summaries

No matplotlib is imported here.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from analysis.utils.io import analysis_dir
from analysis.utils.paths import RESULTS_ROOT
from analysis.utils.results import (FORMAL_CONDITIONS, condition_matrix, mean_and_std,
                            require_runs, select_runs)

CONDITION_LABELS = OrderedDict([
    ("clean", "Clean"), ("downsample", "Downsample"), ("random", "Random"),
    ("segment", "Segment"), ("destructive", "Destructive"),
])
DEGRADED_CONDITIONS = ("downsample", "random", "segment", "destructive")
RECONSTRUCTION_MODELS = OrderedDict([
    ("batterydit", "BatteryDiT"), ("csdi", "CSDI"),
    ("transformer", "Transformer"), ("kalman", "Kalman"),
])
DIRECT_MODELS = OrderedDict([
    ("gru", "GRU"), ("lstm", "LSTM"), ("transformer", "Transformer"),
    ("tcn", "TCN"), ("trb", "TRB"),
])
SEED = 2021


def _seed_run(runs, seed: int = SEED):
    matching = [run for run in runs if run.seed == seed]
    if len(matching) != 1:
        raise FileNotFoundError(f"expected one seed {seed} run, found {[r.seed for r in runs]}")
    return matching[0]


def _trajectory(run, condition: str) -> dict:
    path = run.directory / f"reconstructions_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing full trajectory output: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _reconstruction_runs(dataset: str, results_root: Path):
    return OrderedDict(
        (model, require_runs(select_runs("Reconstruction", dataset, recon_model=model,
                                        results_root=results_root),
                             label, 1 if model == "kalman" else 3))
        for model, label in RECONSTRUCTION_MODELS.items())


def _prediction_runs(dataset: str, results_root: Path):
    direct = OrderedDict(
        (model, require_runs(select_runs("DirectPrediction", dataset, direct_model=model,
                                        results_root=results_root), f"direct {label}", 3))
        for model, label in DIRECT_MODELS.items())
    assisted = OrderedDict(
        (model, require_runs(select_runs("PredictDownstream", dataset, recon_model=model,
                                        predictor="trb", results_root=results_root),
                             f"{label}->TRB", 3))
        for model, label in RECONSTRUCTION_MODELS.items())
    assisted["batterydit_tcn"] = require_runs(
        select_runs("PredictDownstream", dataset, recon_model="batterydit",
                    predictor="tcn", results_root=results_root), "BatteryDiT->TCN", 3)
    return direct, assisted


def main() -> None:
    dataset = "mit"
    recon_runs = _reconstruction_runs(dataset, RESULTS_ROOT)

    # representative cycle per condition: BatteryDiT's median-error cycle
    ours = _seed_run(recon_runs["batterydit"])
    keys, payload = {}, {}
    for condition in DEGRADED_CONDITIONS:
        values = _trajectory(ours, condition)
        errors = np.sqrt(np.mean((values["reconstruction"] - values["clean_voltage"]) ** 2, axis=(1, 2)))
        index = int(np.argmin(np.abs(errors - np.median(errors))))
        key = str(values["keys"][index])
        keys[condition] = key
        for model in RECONSTRUCTION_MODELS:
            archive = _trajectory(_seed_run(recon_runs[model]), condition)
            lookup = {str(k): i for i, k in enumerate(archive["keys"])}
            if key not in lookup:
                raise ValueError(f"representative key {key} absent from {model}/{condition}")
            j = lookup[key]
            for field, source in (("clean", "clean_voltage"), ("observed", "input_voltage"),
                                  ("reconstructed", "reconstruction")):
                payload[f"{model}_{condition}_{field}"] = np.asarray(archive[source])[j].reshape(-1)
    np.savez_compressed(analysis_dir() / "supplementary_representatives.npz", **payload)

    # per-cycle waveform RMSE per model x condition
    waveform = {}
    for column, condition in enumerate(DEGRADED_CONDITIONS):
        for model in RECONSTRUCTION_MODELS:
            archive = _trajectory(_seed_run(recon_runs[model]), condition)
            waveform[f"{model}_{column}"] = np.sqrt(
                np.mean((archive["reconstruction"] - archive["clean_voltage"]) ** 2, axis=(1, 2)))
    np.savez_compressed(analysis_dir() / "supplementary_waveform_rmse.npz", **waveform)

    # Plotting summaries for the legacy SI violin encoding.  Keep the raw
    # cycle-level values above for the violin density, while the explicit
    # IQR/median marks and common log-axis limits are prepared here rather
    # than recomputed by the figure layer.
    quantiles = np.empty((len(RECONSTRUCTION_MODELS), len(DEGRADED_CONDITIONS), 3), dtype=float)
    all_waveform_values = []
    for model_index, model in enumerate(RECONSTRUCTION_MODELS):
        for column, _condition in enumerate(DEGRADED_CONDITIONS):
            series = np.asarray(waveform[f"{model}_{column}"], dtype=float)
            quantiles[model_index, column] = np.quantile(series, [0.25, 0.50, 0.75])
            all_waveform_values.append(series)
    all_waveform_values = np.concatenate(all_waveform_values)
    finite_positive = all_waveform_values[np.isfinite(all_waveform_values) & (all_waveform_values > 0)]
    if len(finite_positive) != len(all_waveform_values):
        raise ValueError("waveform RMSE summaries require finite positive values")
    np.savez_compressed(
        analysis_dir() / "supplementary_waveform_rmse_summary.npz",
        quantiles=quantiles,
        y_min=np.asarray([10 ** np.floor(np.log10(finite_positive.min()))], dtype=float),
        y_max=np.asarray([finite_positive.max()], dtype=float),
        n_values=np.asarray([len(waveform["batterydit_0"])], dtype=int),
    )

    # SOH parity for BatteryDiT->TRB
    assisted_runs = require_runs(select_runs("PredictDownstream", dataset,
                                             recon_model="batterydit", predictor="trb",
                                             results_root=RESULTS_ROOT), "BatteryDiT->TRB", 3)
    parity = {}
    with np.load((_seed_run(assisted_runs).directory / "predictions.npz"),
                 allow_pickle=False) as archive:
        for condition in DEGRADED_CONDITIONS:
            parity[f"{condition}_prediction"] = np.asarray(archive[f"{condition}_prediction"])[:, 0]
            parity[f"{condition}_target"] = np.asarray(archive[f"{condition}_target"])[:, 0]
    np.savez_compressed(analysis_dir() / "supplementary_soh_parity.npz", **parity)

    # Shared limits and error normalization for the legacy 2x2 parity panel.
    # The figure layer uses these frozen values directly so it does not derive
    # axis or colour scales from the plotted observations.
    parity_targets = np.concatenate([
        np.asarray(parity[f"{condition}_target"], dtype=float)
        for condition in DEGRADED_CONDITIONS
    ])
    parity_predictions = np.concatenate([
        np.asarray(parity[f"{condition}_prediction"], dtype=float)
        for condition in DEGRADED_CONDITIONS
    ])
    parity_errors = np.abs(parity_predictions - parity_targets)
    np.savez_compressed(
        analysis_dir() / "supplementary_soh_parity_summary.npz",
        axis_min=np.asarray([min(parity_targets.min(), parity_predictions.min()) - 0.015]),
        axis_max=np.asarray([max(parity_targets.max(), parity_predictions.max()) + 0.015]),
        color_max=np.asarray([np.quantile(parity_errors, 0.95)]),
    )

    # downstream comparison: condition means and per-battery errors
    direct, assisted = _prediction_runs(dataset, RESULTS_ROOT)
    downstream = {}
    for group, runs in (("direct", direct), ("assisted", assisted)):
        for model, items in runs.items():
            mean, std = mean_and_std(condition_matrix(items, "soh_rmse"))
            downstream[f"{group}_{model}_mean"] = mean
            downstream[f"{group}_{model}_sd"] = std
            per_battery = []
            per_battery_by_condition = {condition: [] for condition in FORMAL_CONDITIONS}
            for run in items:
                with np.load(run.directory / "predictions.npz", allow_pickle=False) as archive:
                    for condition in FORMAL_CONDITIONS:
                        prediction = np.asarray(archive[f"{condition}_prediction"])[:, 0]
                        target = np.asarray(archive[f"{condition}_target"])[:, 0]
                        key_name = f"{condition}_keys"
                        if key_name in archive.files:
                            keys_arr = np.asarray(archive[key_name]).astype(str)
                        else:
                            # DirectPrediction archives omit keys; the clean
                            # reconstruction archive defines the same cycle order
                            fallback = _seed_run(recon_runs["batterydit"], run.seed)
                            keys_arr = np.asarray(
                                _trajectory(fallback, condition)["keys"]).astype(str)
                        if len(keys_arr) != len(prediction):
                            raise ValueError(
                                f"{run.directory} key/prediction count mismatch for {condition}")
                        cells = np.asarray([k.split("/", 1)[0] for k in keys_arr])
                        condition_errors = [
                            float(np.sqrt(np.mean((prediction[cells == cell] - target[cells == cell]) ** 2)))
                            for cell in sorted(set(cells))
                        ]
                        per_battery.extend(condition_errors)
                        per_battery_by_condition[condition].extend(condition_errors)
            downstream[f"{group}_{model}_battery_errors"] = np.asarray(per_battery)
            for condition, errors in per_battery_by_condition.items():
                downstream[f"{group}_{model}_{condition}_battery_errors"] = np.asarray(errors)
    np.savez_compressed(analysis_dir() / "supplementary_downstream.npz", **downstream)

    # Plotting products for the legacy 2x2 direct/assisted comparison.  The
    # figure layer receives condition-grouped errors and precomputed IQR/
    # median summaries, rather than reconstructing them from concatenated
    # per-battery vectors.
    downstream_summary = {}
    bar_ymax, violin_ymax = [], []
    for group, models in (("direct", direct), ("assisted", assisted)):
        bar_values, violin_values = [], []
        for model in models:
            bar_values.append(np.asarray(downstream[f"{group}_{model}_mean"], dtype=float))
            grouped = []
            quantiles = []
            for condition in FORMAL_CONDITIONS:
                series = np.asarray(
                    downstream[f"{group}_{model}_{condition}_battery_errors"], dtype=float)
                grouped.append(series)
                quantiles.append(np.quantile(series, [0.25, 0.50, 0.75]))
            downstream_summary[f"{group}_{model}_quantiles"] = np.asarray(quantiles, dtype=float)
            violin_values.extend(grouped)
        bar_ymax.append(float(np.max(np.concatenate(bar_values)) * 1.18))
        violin_ymax.append(float(np.max(np.concatenate(violin_values)) * 1.08))
    downstream_summary["bar_ymax"] = np.asarray(bar_ymax, dtype=float)
    downstream_summary["violin_ymax"] = np.asarray(violin_ymax, dtype=float)
    np.savez_compressed(
        analysis_dir() / "supplementary_downstream_summary.npz",
        **downstream_summary,
    )
    print(f"wrote supplementary arrays -> {analysis_dir()}")


if __name__ == "__main__":
    main()
