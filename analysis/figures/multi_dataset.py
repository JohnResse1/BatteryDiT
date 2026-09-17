"""Draw cross-system trajectory recovery and downstream diagnostic utility."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from common.paths import AUDIT_ROOT, RESULTS_ROOT, create_output_directories
from common.plotting import COLORS, FIGURE_WIDTH, apply_paper_style, export_figure
from common.results import FORMAL_CONDITIONS, condition_matrix, require_runs, select_runs


DATASETS = OrderedDict([
    ("mit", "MIT"),
    ("hust", "HUST"),
    ("mich", "MICH"),
    ("sdu", "SDU"),
    ("stanford_2", "Stanford"),
    ("tju_nca_45c", "TJU-NCA"),
    ("tju_ncm_45c", "TJU-NCM"),
])
CHEMISTRIES = {
    "mit": "LFP", "hust": "LFP", "mich": "NMC111", "sdu": "NMC532",
    "stanford_2": "NMC532", "tju_nca_45c": "NCA", "tju_ncm_45c": "NCM",
}
DEGRADED_CONDITIONS = OrderedDict([
    ("downsample", "Downsample"),
    ("random", "Random"),
    ("segment", "Segment"),
    ("destructive", "Destructive"),
])
SEED_FOR_INPUT = 2021
MAX_CANDIDATES_PER_BATTERY = 64


def _runs(dataset: str, results_root: Path):
    reconstruction = require_runs(
        select_runs(
            "multi_dataset", dataset, recon_model="batterydit",
            task_name="Reconstruction", results_root=results_root,
        ),
        f"{dataset} multi-dataset reconstruction", 3,
    )
    prediction = require_runs(
        select_runs(
            "multi_dataset", dataset, recon_model="batterydit", predictor="trb",
            task_name="PredictDownstream", results_root=results_root,
        ),
        f"{dataset} multi-dataset downstream prediction", 3,
    )
    return reconstruction, prediction


def _trajectory_path(run, condition: str) -> Path:
    path = run.directory / f"reconstructions_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing full trajectory output: {path}; rerun reconstruction with "
            "--save_reconstructions true"
        )
    return path


def _cells(keys: np.ndarray) -> np.ndarray:
    return np.asarray([str(key).split("/", 1)[0] for key in keys])


def _candidate_indices(cells: np.ndarray) -> np.ndarray:
    """Take at most 64 evenly spaced cycles from each battery."""
    selected: list[int] = []
    for cell in dict.fromkeys(cells.astype(str)):
        members = np.flatnonzero(cells == cell)
        count = min(len(members), MAX_CANDIDATES_PER_BATTERY)
        positions = np.unique(np.rint(np.linspace(0, len(members) - 1, count)).astype(int))
        selected.extend(members[positions].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _representative(runs) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Select one segment-missing cycle by a battery-balanced median rule."""
    condition = "segment"
    first_path = _trajectory_path(next(run for run in runs if run.seed == SEED_FOR_INPUT), condition)
    with np.load(first_path, allow_pickle=False) as archive:
        required = {"keys", "clean_voltage", "input_voltage", "reconstruction"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{first_path} is missing {sorted(missing)}")
        keys = np.asarray(archive["keys"]).astype(str)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{first_path} contains duplicate sample keys")
    candidates = _candidate_indices(_cells(keys))
    seed_errors, candidate_reconstructions = [], []
    candidate_clean = candidate_input = None
    for run in runs:
        path = _trajectory_path(run, condition)
        with np.load(path, allow_pickle=False) as archive:
            run_keys = np.asarray(archive["keys"]).astype(str)
            if not np.array_equal(run_keys, keys):
                raise ValueError(f"trajectory key order differs across seeds: {path}")
            clean = np.asarray(archive["clean_voltage"], dtype=np.float32)
            reconstruction = np.asarray(archive["reconstruction"], dtype=np.float32)
            if clean.shape != reconstruction.shape or len(clean) != len(keys):
                raise ValueError(f"invalid paired trajectory shapes in {path}")
            error = np.sqrt(np.mean((reconstruction - clean) ** 2, axis=(1, 2)))
            seed_errors.append(error[candidates])
            candidate_reconstructions.append(reconstruction[candidates, :, 0])
            if run.seed == SEED_FOR_INPUT:
                candidate_clean = clean[candidates, :, 0]
                candidate_input = np.asarray(archive["input_voltage"], dtype=np.float32)[candidates, :, 0]
    if candidate_clean is None or candidate_input is None:
        raise FileNotFoundError(f"seed {SEED_FOR_INPUT} trajectory is required for representative input")
    score = np.stack(seed_errors).mean(axis=0)
    local_index = int(np.argmin(np.abs(score - np.median(score))))
    item = {
        "clean": candidate_clean[local_index],
        "input": candidate_input[local_index],
        "reconstruction": np.stack(candidate_reconstructions).mean(axis=0)[local_index],
        "observed": np.abs(candidate_input[local_index]) > 1e-12,
    }
    metadata = {
        "key": str(keys[candidates[local_index]]),
        "candidate_count": int(len(candidates)),
        "selection_score_v": float(score[local_index]),
        "candidate_median_score_v": float(np.median(score)),
        "selection_rule": (
            "up to 64 evenly spaced cycles per battery; nearest to the median "
            "three-seed mean waveform RMSE"
        ),
    }
    return item, metadata


def _summaries(results_root: Path):
    reconstruction_mean, reconstruction_std = [], []
    prediction_delta_mean, prediction_delta_std = [], []
    corrupted_rmse, representatives, metadata = [], {}, {}
    columns = [FORMAL_CONDITIONS.index(name) for name in DEGRADED_CONDITIONS]
    for dataset in DATASETS:
        recon_runs, prediction_runs = _runs(dataset, results_root)
        recon_values = condition_matrix(recon_runs, "battery_balanced_rmse")[:, columns]
        prediction_values = condition_matrix(prediction_runs, "soh_rmse")
        deltas = prediction_values[:, columns] - prediction_values[:, [FORMAL_CONDITIONS.index("clean")]]
        reconstruction_mean.append(recon_values.mean(axis=0))
        reconstruction_std.append(recon_values.std(axis=0, ddof=1))
        prediction_delta_mean.append(deltas.mean(axis=0))
        prediction_delta_std.append(deltas.std(axis=0, ddof=1))

        input_run = next(run for run in recon_runs if run.seed == SEED_FOR_INPUT)
        condition_errors = []
        for condition in DEGRADED_CONDITIONS:
            path = _trajectory_path(input_run, condition)
            with np.load(path, allow_pickle=False) as archive:
                corrupted = np.asarray(archive["input_voltage"], dtype=np.float32)
                clean = np.asarray(archive["clean_voltage"], dtype=np.float32)
            if corrupted.shape != clean.shape:
                raise ValueError(f"input/clean shape mismatch in {path}")
            condition_errors.append(float(np.sqrt(np.mean((corrupted - clean) ** 2, axis=(1, 2))).mean()))
        corrupted_rmse.append(condition_errors)
        representatives[dataset], metadata[dataset] = _representative(recon_runs)
    return (
        np.asarray(corrupted_rmse),
        np.asarray(reconstruction_mean), np.asarray(reconstruction_std),
        np.asarray(prediction_delta_mean), np.asarray(prediction_delta_std),
        representatives, metadata,
    )


def _missing_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def draw(dataset: str = "all", *, results_root: Path = RESULTS_ROOT) -> dict[str, str]:
    """Build the three-panel cross-system figure from formal seven-cohort runs."""
    del dataset
    apply_paper_style()
    (
        corrupted, reconstructed, reconstructed_std,
        delta_mean, delta_std, representatives, metadata,
    ) = _summaries(Path(results_root))

    figure = plt.figure(figsize=(FIGURE_WIDTH, 166 / 25.4))
    outer = figure.add_gridspec(
        3, 1, height_ratios=(0.83, 1.15, 0.92),
        left=0.075, right=0.985, bottom=0.075, top=0.965, hspace=0.46,
    )

    curve_grid = outer[0].subgridspec(1, len(DATASETS), wspace=0.42)
    curve_axes = []
    for column, (name, label) in enumerate(DATASETS.items()):
        axis = figure.add_subplot(curve_grid[0, column])
        curve_axes.append(axis)
        item = representatives[name]
        clean, recovery, observed = item["clean"], item["reconstruction"], item["observed"]
        x = np.linspace(0.0, 1.0, len(clean))
        for start, end in _missing_spans(~observed):
            axis.axvspan(x[start], x[end], color="#D8D8D8", alpha=0.42, linewidth=0)
        axis.plot(x, clean, color="#111111", linewidth=0.65)
        axis.plot(x, recovery, color=COLORS["batterydit"], linewidth=0.95)
        low, high = min(float(clean.min()), float(recovery.min())), max(float(clean.max()), float(recovery.max()))
        pad = max((high - low) * 0.07, 0.02)
        axis.set(xlim=(0, 1), ylim=(low - pad, high + pad), xticks=(0, 1), title=label)
        axis.text(0.5, 1.012, CHEMISTRIES[name], transform=axis.transAxes, ha="center", va="bottom", fontsize=4.8, color="#5A5A5A")
        axis.grid(alpha=0.08, linewidth=0.3)
        if column == 0:
            axis.set_ylabel("Voltage (V)")
    curve_axes[0].legend(handles=[
        Line2D([], [], color="#111111", label="Clean"),
        Line2D([], [], color=COLORS["batterydit"], label="BatteryDiT"),
        Patch(facecolor="#D8D8D8", edgecolor="none", alpha=0.6, label="Missing"),
    ], loc="lower right", fontsize=5.0)

    recovery_grid = outer[1].subgridspec(1, len(DEGRADED_CONDITIONS), wspace=0.18)
    recovery_axes = []
    y = np.arange(len(DATASETS))
    ticks = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    for column, (_, label) in enumerate(DEGRADED_CONDITIONS.items()):
        axis = figure.add_subplot(recovery_grid[0, column], sharex=recovery_axes[0] if recovery_axes else None)
        recovery_axes.append(axis)
        for row in range(len(DATASETS)):
            axis.plot([corrupted[row, column], reconstructed[row, column]], [y[row], y[row]], color="#C7C7C7", linewidth=0.75)
        axis.scatter(corrupted[:, column], y, s=13, color="#A6A6A6", edgecolor="white", linewidth=0.3, zorder=2)
        axis.errorbar(reconstructed[:, column], y, xerr=reconstructed_std[:, column], fmt="o", ms=3.2, mfc=COLORS["batterydit"], mec="white", mew=0.3, ecolor=COLORS["batterydit"], elinewidth=0.55, capsize=1.25, zorder=3)
        axis.set_xscale("log")
        axis.set_xlim(1e-5, 5.0)
        axis.set_xticks(ticks)
        axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_ylim(len(DATASETS) - 0.45, -0.55)
        axis.set_yticks(y)
        axis.set_yticklabels(DATASETS.values() if column == 0 else [])
        axis.set_title(label)
        axis.set_xlabel("Waveform RMSE (V)")
        axis.grid(axis="x", alpha=0.08, linewidth=0.3)
    recovery_axes[-1].legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#A6A6A6", label="Corrupted"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["batterydit"], label="BatteryDiT"),
    ], loc="lower right")

    axis_c = figure.add_subplot(outer[2])
    colors = ("#2166AC", "#67A1CB", "#EAA88C", "#D6604D")
    markers = ("o", "s", "^", "D")
    offsets = np.asarray((-0.18, -0.06, 0.06, 0.18))
    for column, ((_, label), color, marker) in enumerate(zip(DEGRADED_CONDITIONS.items(), colors, markers)):
        axis_c.errorbar(delta_mean[:, column], y + offsets[column], xerr=delta_std[:, column], fmt=marker, ms=3.4, mfc=color, mec="white", mew=0.3, ecolor=color, elinewidth=0.55, capsize=1.3, label=label)
    axis_c.axvline(0.0, color="#666666", linewidth=0.65, linestyle="--")
    axis_c.set_ylim(len(DATASETS) - 0.55, -0.55)
    axis_c.set_yticks(y)
    axis_c.set_yticklabels(DATASETS.values())
    axis_c.set_xlabel(r"Residual diagnostic penalty, $\Delta E_{\rm diag}=E_{\rm reconstructed}-E_{\rm clean}$")
    axis_c.legend(loc="lower right", ncol=4)
    axis_c.grid(axis="x", alpha=0.08, linewidth=0.3)

    panel_x = 0.012
    for label, axis in (("a", curve_axes[0]), ("b", recovery_axes[0]), ("c", axis_c)):
        figure.text(panel_x, axis.get_position().y1 + 0.014, f"({label})", fontsize=8, fontweight="bold", va="bottom")

    outputs = export_figure(figure, "multi_dataset")
    plt.close(figure)
    create_output_directories()
    (AUDIT_ROOT / "multi_dataset_representatives.json").write_text(
        json.dumps({"condition": "segment", "seeds": [2021, 2022, 2023], "datasets": metadata}, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs
