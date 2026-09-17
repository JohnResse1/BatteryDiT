"""Supplementary benchmark figures from complete trajectory/prediction outputs."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.paths import RESULTS_ROOT
from common.plotting import COLORS, FIGURE_WIDTH, apply_paper_style, export_figure, panel_label
from common.results import FORMAL_CONDITIONS, condition_matrix, mean_and_std, require_runs, select_runs
from figures.benchmark import CONDITION_LABELS, DEGRADED_CONDITIONS, DIRECT_MODELS, RECONSTRUCTION_MODELS, _seed_run, _trajectory


def _reconstruction_runs(dataset: str, root: Path):
    return OrderedDict((model, require_runs(select_runs("Reconstruction", dataset, recon_model=model, results_root=root), label, 1 if model == "kalman" else 3)) for model, label in RECONSTRUCTION_MODELS.items())


def _representative_keys(runs) -> dict[str, str]:
    ours = _seed_run(runs["batterydit"])
    selected = {}
    for condition in DEGRADED_CONDITIONS:
        values = _trajectory(ours, condition)
        errors = np.sqrt(np.mean((values["reconstruction"] - values["clean_voltage"]) ** 2, axis=(1, 2)))
        selected[condition] = str(values["keys"][int(np.argmin(np.abs(errors - np.median(errors))))])
    return selected


def draw_reconstruction_examples(dataset: str = "mit", *, results_root: Path = RESULTS_ROOT):
    apply_paper_style(); root = Path(results_root)
    runs = _reconstruction_runs(dataset, root); keys = _representative_keys(runs)
    figure, axes = plt.subplots(4, 4, figsize=(FIGURE_WIDTH, 145 / 25.4), sharex=True, sharey=True)
    for row, condition in enumerate(DEGRADED_CONDITIONS):
        for column, (model, label) in enumerate(RECONSTRUCTION_MODELS.items()):
            axis = axes[row, column]; values = _trajectory(_seed_run(runs[model]), condition)
            lookup = {str(key): index for index, key in enumerate(values["keys"])}
            index = lookup[keys[condition]]; x = np.arange(values["clean_voltage"].shape[1])
            observed = values["input_voltage"][index, :, 0]; valid = np.abs(observed) > 1e-12
            axis.plot(x, values["clean_voltage"][index, :, 0], color="#111111", label="Clean")
            axis.plot(x[valid], observed[valid], color="#999999", linewidth=0.65, label="Observed")
            axis.plot(x, values["reconstruction"][index, :, 0], color=COLORS[model], label="Reconstructed")
            axis.grid(alpha=0.08, linewidth=0.3)
            if row == 0: axis.set_title(label)
            if column == 0: axis.set_ylabel(f"{CONDITION_LABELS[condition]}\nVoltage (V)")
            if row == 3: axis.set_xlabel("Charging step")
    axes[0, 0].legend(fontsize=4.5)
    figure.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.07, wspace=0.10, hspace=0.16)
    outputs = export_figure(figure, "supplementary_reconstruction_examples", supplementary=True); plt.close(figure); return outputs


def draw_waveform_distributions(dataset: str = "mit", *, results_root: Path = RESULTS_ROOT):
    apply_paper_style(); runs = _reconstruction_runs(dataset, Path(results_root))
    figure, axes = plt.subplots(1, 4, figsize=(FIGURE_WIDTH, 53 / 25.4), sharey=True)
    for column, condition in enumerate(DEGRADED_CONDITIONS):
        values = []
        for model in RECONSTRUCTION_MODELS:
            archive = _trajectory(_seed_run(runs[model]), condition)
            values.append(np.sqrt(np.mean((archive["reconstruction"] - archive["clean_voltage"]) ** 2, axis=(1, 2))))
        violin = axes[column].violinplot(values, positions=np.arange(4), showmedians=True, showextrema=False)
        for body, model in zip(violin["bodies"], RECONSTRUCTION_MODELS): body.set_facecolor(COLORS[model]); body.set_alpha(0.65)
        axes[column].set_title(CONDITION_LABELS[condition]); axes[column].set_yscale("log")
        axes[column].set_xticks(np.arange(4)); axes[column].set_xticklabels(RECONSTRUCTION_MODELS.values(), rotation=30, ha="right")
        axes[column].grid(axis="y", alpha=0.12)
    axes[0].set_ylabel("Cycle waveform RMSE (V)"); panel_label(axes[0], "a")
    figure.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.24, wspace=0.10)
    outputs = export_figure(figure, "supplementary_waveform_distributions", supplementary=True); plt.close(figure); return outputs


def _prediction_archive(run) -> dict[str, np.ndarray]:
    path = run.directory / "predictions.npz"
    if not path.is_file(): raise FileNotFoundError(f"missing prediction arrays: {path}")
    with np.load(path, allow_pickle=False) as archive: return {name: np.asarray(archive[name]) for name in archive.files}


def draw_soh_parity(dataset: str = "mit", *, results_root: Path = RESULTS_ROOT):
    apply_paper_style(); runs = require_runs(select_runs("PredictDownstream", dataset, recon_model="batterydit", predictor="trb", results_root=Path(results_root)), "BatteryDiT→TRB", 3)
    values = _prediction_archive(_seed_run(runs)); figure, axes = plt.subplots(1, 4, figsize=(FIGURE_WIDTH, 48 / 25.4), sharex=True, sharey=True)
    for column, condition in enumerate(DEGRADED_CONDITIONS):
        prediction = values[f"{condition}_prediction"][:, 0]; target = values[f"{condition}_target"][:, 0]; error = np.abs(prediction - target)
        axes[column].scatter(target, prediction, c=error, cmap="magma_r", s=4, alpha=0.45, linewidth=0)
        low, high = min(target.min(), prediction.min()), max(target.max(), prediction.max()); axes[column].plot([low, high], [low, high], color="#D6604D", linestyle="--", linewidth=0.8)
        axes[column].set_title(CONDITION_LABELS[condition]); axes[column].set_xlabel("True SOH"); axes[column].grid(alpha=0.08)
    axes[0].set_ylabel("Predicted SOH"); panel_label(axes[0], "a")
    figure.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.18, wspace=0.10)
    outputs = export_figure(figure, "supplementary_soh_parity", supplementary=True); plt.close(figure); return outputs


def _prediction_runs(dataset: str, root: Path):
    direct = OrderedDict((model, require_runs(select_runs("DirectPrediction", dataset, direct_model=model, results_root=root), f"direct {label}", 3)) for model, label in DIRECT_MODELS.items())
    assisted = OrderedDict((model, require_runs(select_runs("PredictDownstream", dataset, recon_model=model, predictor="trb", results_root=root), f"{label}→TRB", 3)) for model, label in RECONSTRUCTION_MODELS.items())
    assisted["batterydit_tcn"] = require_runs(select_runs("PredictDownstream", dataset, recon_model="batterydit", predictor="tcn", results_root=root), "BatteryDiT→TCN", 3)
    return direct, assisted


def _battery_errors(runs, reconstruction_runs) -> list[np.ndarray]:
    pooled = []
    for run in runs:
        values = _prediction_archive(run); per_battery = []
        for condition in FORMAL_CONDITIONS:
            prediction, target = values[f"{condition}_prediction"][:, 0], values[f"{condition}_target"][:, 0]
            key_name = f"{condition}_keys"
            keys = values[key_name].astype(str) if key_name in values else _trajectory(_seed_run(reconstruction_runs, run.seed), condition)["keys"].astype(str)
            cells = np.asarray([key.split("/", 1)[0] for key in keys])
            per_battery.extend(float(np.sqrt(np.mean((prediction[cells == cell] - target[cells == cell]) ** 2))) for cell in sorted(set(cells)))
        pooled.append(np.asarray(per_battery))
    return pooled


def draw_downstream_comparison(dataset: str = "mit", *, results_root: Path = RESULTS_ROOT):
    apply_paper_style(); root = Path(results_root); direct, assisted = _prediction_runs(dataset, root)
    recon_runs = require_runs(select_runs("Reconstruction", dataset, recon_model="batterydit", results_root=root), "BatteryDiT reconstruction", 3)
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 105 / 25.4))
    groups = ((direct, "Direct prediction"), (assisted, "Reconstruction-assisted"))
    for column, (runs, title) in enumerate(groups):
        x=np.arange(5); width=.78/len(runs)
        for index,(model,items) in enumerate(runs.items()):
            mean,std=mean_and_std(condition_matrix(items,"soh_rmse")); color=COLORS.get(model.split("_")[0],"#777777")
            axes[0,column].bar(x+(index-(len(runs)-1)/2)*width,mean,width,yerr=std,capsize=1,color=color,alpha=.7,label=model.replace("batterydit_tcn","BatteryDiT→TCN"))
        axes[0,column].set_title(title); axes[0,column].set_xticks(x); axes[0,column].set_xticklabels([CONDITION_LABELS[c] for c in FORMAL_CONDITIONS],rotation=28,ha="right"); axes[0,column].set_ylabel("SOH RMSE"); axes[0,column].legend(ncol=2,fontsize=4.2)
        battery_values=[]; labels=[]
        for model,items in runs.items():
            model_values=[]
            for run in items:
                fallback=[candidate for candidate in recon_runs if candidate.seed==run.seed]
                model_values.extend(_battery_errors([run],fallback))
            battery_values.append(np.concatenate(model_values)); labels.append(model.replace("batterydit_tcn","BatteryDiT→TCN"))
        axes[1,column].violinplot(battery_values,positions=np.arange(len(labels)),showmedians=True,showextrema=False)
        axes[1,column].set_xticks(np.arange(len(labels))); axes[1,column].set_xticklabels(labels,rotation=28,ha="right"); axes[1,column].set_ylabel("Battery-condition SOH RMSE"); axes[1,column].grid(axis="y",alpha=.12)
        panel_label(axes[0,column], "a" if column==0 else "b"); panel_label(axes[1,column], "c" if column==0 else "d")
    figure.subplots_adjust(left=.08,right=.99,top=.95,bottom=.12,wspace=.25,hspace=.55)
    outputs=export_figure(figure,"supplementary_downstream_comparison",supplementary=True);plt.close(figure);return outputs
