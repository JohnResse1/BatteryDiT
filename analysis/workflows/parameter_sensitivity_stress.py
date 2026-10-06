"""Complete R3 Pop--Ind parameter sensitivity over the frozen stress grid.

The earlier ``parameter_sensitivity`` workflow evaluates clean trajectories
only.  This workflow preserves the old one-factor-at-a-time 11-setting grid
and evaluates every setting on the formal six-system, five-corruption, L1--L5
stress protocol.

The Population--Individual geometry is refit from clean train batteries for
each setting.  Reconstruction is performed only with the checkpoint already
registered for the corresponding formal system; no checkpoint is trained or
updated.  The resulting statistics are analysis outputs and are aggregated
cycle -> battery-condition -> system-condition.
"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from analysis.utils.paths import REPOSITORY_ROOT, RESULTS_ROOT

from . import _certified as C
from .provenance import canonical_run


BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261004
EVALUATION_SEED = 3407
EVAL_BATCH_SIZE = 128

DISPLAY_CORRUPTION = {
    "downsample": "Downsample",
    "random_missing": "Random",
    "segment_missing": "Segment",
    "destructive": "Destructive",
    "hybrid": "Hybrid",
}


@dataclass(frozen=True)
class Setting:
    setting_id: str
    parameter: str
    value: str
    k_ref: int = 5
    degree: int = 3
    fixed_knots: int | None = None
    alpha_multiplier: float = 1.0
    grid_size: int = 500
    canonical: bool = False


SETTINGS = (
    Setting("canonical", "canonical", "canonical", canonical=True),
    Setting("K_ref_3", "K_ref", "3", k_ref=3),
    Setting("K_ref_7", "K_ref", "7", k_ref=7),
    Setting("degree_2", "spline_degree", "2", degree=2),
    Setting("knots_4", "n_knots", "4", fixed_knots=4),
    Setting("knots_5", "n_knots", "5", fixed_knots=5),
    Setting("knots_6", "n_knots", "6", fixed_knots=6),
    Setting("alpha_0.1x", "ridge_alpha", "0.1x_canonical", alpha_multiplier=0.1),
    Setting("alpha_10x", "ridge_alpha", "10x_canonical", alpha_multiplier=10.0),
    Setting("grid_200", "projection_grid", "200", grid_size=200),
    Setting("grid_1000", "projection_grid", "1000", grid_size=1000),
)


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    return float(spearmanr(x[keep], y[keep]).statistic)


def battery_values(values: np.ndarray, cells: np.ndarray) -> pd.Series:
    return pd.DataFrame({"battery_id": np.asarray(cells).astype(str), "value": values}).groupby(
        "battery_id", sort=True
    ).value.mean()


def bootstrap_mean_difference(values: np.ndarray, seed_offset: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(seed_offset))
    draw = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    estimate = values[draw].mean(axis=1)
    return float(np.percentile(estimate, 2.5)), float(np.percentile(estimate, 97.5))


def key_parts(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells, cycles = [], []
    for key in np.asarray(keys).astype(str):
        cell, cycle = key.rsplit("/", 1)
        cells.append(cell)
        cycles.append(int(cycle.removesuffix(".npz")))
    return np.asarray(cells), np.asarray(cycles, dtype=np.int32)


def load_test_with_current(dataset: str) -> dict[str, np.ndarray]:
    """Load the certified clean test order and the current needed for stress input."""
    from data_provider.data_loader import BatteryDataset

    source = BatteryDataset(dataset, split="test", condition="clean")
    voltage, current, soh, keys = [], [], [], []
    for path in source.paths:
        with np.load(path, allow_pickle=False) as sample:
            voltage.append(np.asarray(sample["V"], dtype=np.float32))
            current.append(np.asarray(sample["I"], dtype=np.float32))
            soh.append(float(np.asarray(sample["SOH"], dtype=np.float32).reshape(-1)[0]))
        keys.append(str(path.relative_to(source.root)))
    key_array = np.asarray(keys)
    cells, cycles = key_parts(key_array)
    return {
        "voltage": np.stack(voltage),
        "current": np.stack(current),
        "soh": np.asarray(soh, dtype=np.float32),
        "keys": key_array,
        "cells": cells,
        "cycles": cycles,
    }


def residualize(
    data: dict[str, np.ndarray], k_ref: int
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    voltage = np.asarray(data["voltage"], dtype=np.float32)
    cells = np.asarray(data["cells"]).astype(str)
    cycles = np.asarray(data["cycles"], dtype=np.int32)
    residual = np.full_like(voltage, np.nan)
    references: dict[str, np.ndarray] = {}
    keep = np.zeros(len(cells), dtype=bool)
    for cell in sorted(set(cells)):
        index = np.where(cells == cell)[0]
        index = index[np.argsort(cycles[index], kind="stable")]
        if len(index) < k_ref:
            raise ValueError(f"{cell}: {len(index)} cycles cannot support K_ref={k_ref}")
        reference = np.median(voltage[index[:k_ref]], axis=0).astype(np.float32)
        references[cell] = reference
        residual[index] = voltage[index] - reference
        keep[index[k_ref:]] = True
    return residual, references, keep


def fit_model(
    soh: np.ndarray,
    residual: np.ndarray,
    knots: int,
    degree: int,
    alpha: float,
):
    spline = SplineTransformer(
        n_knots=int(knots),
        degree=int(degree),
        knots="uniform",
        include_bias=False,
        extrapolation="constant",
    )
    design = spline.fit_transform(np.asarray(soh, dtype=np.float64).reshape(-1, 1))
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="cholesky").fit(
        design, np.asarray(residual, dtype=np.float32).reshape(len(residual), -1)
    )
    return spline, model


def predict(model, spline, soh: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.predict(spline.transform(np.asarray(soh, dtype=np.float64).reshape(-1, 1))),
        dtype=np.float32,
    )


class Projection:
    """Low-rank exact nearest-point search over a spline/Ridge population grid."""

    def __init__(self, spline, model, soh_min: float, soh_max: float, grid_size: int):
        self.grid_soh = np.linspace(float(soh_min), float(soh_max), int(grid_size))
        self.grid_population = predict(model, spline, self.grid_soh).astype(np.float64)
        span = np.vstack([
            np.asarray(model.intercept_, dtype=np.float64)[None, :],
            np.asarray(model.coef_, dtype=np.float64).T,
        ])
        self.basis, _ = np.linalg.qr(span.T, mode="reduced")
        self.grid_coordinates = self.grid_population @ self.basis
        self.grid_norm = np.sum(self.grid_coordinates ** 2, axis=1)

    def nearest(self, trajectories: np.ndarray, chunk_size: int = 256):
        trajectories = np.asarray(trajectories, dtype=np.float64)
        chosen = np.empty(len(trajectories), dtype=np.int32)
        for start in range(0, len(trajectories), chunk_size):
            stop = min(start + chunk_size, len(trajectories))
            coordinates = trajectories[start:stop] @ self.basis
            distance = self.grid_norm[None, :] - 2.0 * (
                coordinates @ self.grid_coordinates.T
            )
            chosen[start:stop] = np.argmin(distance, axis=1)
        return self.grid_soh[chosen], self.grid_population[chosen]

    def audit(self, trajectories: np.ndarray) -> None:
        sample = np.asarray(trajectories[: min(8, len(trajectories))], dtype=np.float64)
        direct = np.sum(self.grid_population ** 2, axis=1)[None, :] - 2.0 * (
            sample @ self.grid_population.T
        )
        reduced_soh, _ = self.nearest(sample, chunk_size=len(sample))
        reduced = np.searchsorted(self.grid_soh, reduced_soh)
        if not np.array_equal(np.argmin(direct, axis=1), reduced):
            raise RuntimeError("low-rank population projection audit failed")


def setting_row(setting: Setting, canonical_knots: int, canonical_alpha: float) -> dict:
    return {
        "setting_id": setting.setting_id,
        "parameter": setting.parameter,
        "value": setting.value,
        "is_canonical": setting.canonical,
        "K_ref": setting.k_ref,
        "spline_degree": setting.degree,
        "n_knots": setting.fixed_knots or canonical_knots,
        "ridge_alpha": canonical_alpha * setting.alpha_multiplier,
        "projection_grid": setting.grid_size,
    }


def aggregate_condition(battery: pd.DataFrame, base: dict) -> list[dict]:
    rows = []
    for (corruption, severity, severity_index), group in battery.groupby(
        ["corruption", "severity", "severity_index"], sort=False
    ):
        row = dict(base)
        row.update({
            "corruption": corruption,
            "severity": severity,
            "severity_index": int(severity_index),
            "n_batteries": int(group.battery_id.nunique()),
            "n_cycles": int(group.n_cycles.sum()),
        })
        for metric in ("E_pop", "E_ind", "Q", "C_ind"):
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"median_{metric}"] = float(np.median(values))
            row[f"IQR_{metric}_low"] = float(np.percentile(values, 25))
            row[f"IQR_{metric}_high"] = float(np.percentile(values, 75))
        rows.append(row)
    return rows


def reconstruction_result_directory(dataset: str, seed: int) -> Path:
    group = "Reconstruction" if dataset == "mit" else "multi_dataset"
    return RESULTS_ROOT / group / dataset / f"Reconstruction_seed{seed}_batterydit_{dataset}_0"


def checkpoint_path(dataset: str, seed: int) -> Path:
    return (
        REPOSITORY_ROOT / "source_code" / "checkpoints" / "Reconstruction" / dataset
        / f"Reconstruction_seed{seed}_batterydit_{dataset}_0" / "checkpoint.pth"
    )


def build_reconstruction_model(dataset: str, seed: int, device: torch.device):
    """Build and strictly load the already-frozen formal BatteryDiT checkpoint."""
    from models.BatteryDiT import Model
    from models.checkpoint_loading import load_reconstruction_checkpoint

    result_dir = reconstruction_result_directory(dataset, seed)
    config_path = result_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    arguments = json.loads(config_path.read_text())["arguments"]
    args = SimpleNamespace(**arguments)
    args.data = dataset
    args.gpu = 0
    args.use_gpu = device.type == "cuda"
    args.sampling_steps = 1
    args.ocv_table_path = str(
        REPOSITORY_ROOT / "source_code" / "models" / "ocv" / f"{dataset}.npz"
    )
    model = Model(args).float().to(device)
    path = checkpoint_path(dataset, seed)
    load_reconstruction_checkpoint(model, path, map_location=device, strict=True)
    model.eval()
    return model, args, path


def generate_stress_reconstruction(
    model,
    test: dict[str, np.ndarray],
    condition: str,
    level_index: int,
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
) -> np.ndarray:
    """Reproduce one formal condition using the registered deterministic protocol."""
    from utils.stress_corruptions import apply_stress_corruption, reset_stress_rng

    reset_stress_rng(EVALUATION_SEED, condition, level_index)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(test["voltage"]), int(batch_size)):
            stop = min(start + int(batch_size), len(test["voltage"]))
            voltage = torch.from_numpy(test["voltage"][start:stop]).to(device)
            current = torch.from_numpy(test["current"][start:stop]).to(device)
            corrupted_voltage, corrupted_current = apply_stress_corruption(
                condition, level_index, voltage, current
            )
            normalized = model(
                V=corrupted_voltage,
                I=corrupted_current,
                mode="recon_only_sampling",
                steps=1,
            )
            outputs.append(model._denormalize_phy(normalized).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def reconstruction_metrics(
    reconstructed: np.ndarray,
    test: dict[str, np.ndarray],
    state: dict,
) -> pd.DataFrame:
    eligible = state["eligible"]
    reconstructed_deviation = np.asarray(reconstructed, dtype=np.float32)[eligible].reshape(
        eligible.sum(), -1
    ) - state["reference"]
    h_rec, population = state["projection"].nearest(reconstructed_deviation)
    individual = reconstructed_deviation - population
    clean_individual = state["clean_individual"]
    clean_norm = state["clean_norm"]
    e_ind = np.sqrt(np.mean((individual - clean_individual) ** 2, axis=1)) / state["s_ind"]
    q = np.sqrt(np.sum(individual ** 2, axis=1) / clean_norm)
    a_ind = np.sum(individual * clean_individual, axis=1) / clean_norm
    frame = pd.DataFrame({
        "battery_id": np.asarray(test["cells"])[eligible].astype(str),
        "cycle_id": np.asarray(test["cycles"])[eligible].astype(int),
        "true_soh": np.asarray(test["soh"])[eligible],
        "E_pop": np.abs(h_rec - np.asarray(test["soh"])[eligible]),
        "E_ind": e_ind,
        "Q": q,
        "C_ind": a_ind / q,
    })
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("non-finite parameter-sensitivity metric")
    return frame


def prepare_state(
    setting: Setting,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    canonical_knots: int,
    canonical_alpha: float,
    canonical_eligible: np.ndarray,
) -> tuple[dict, dict]:
    train_residual, _, train_keep = residualize(train, setting.k_ref)
    validation_residual, _, validation_keep = residualize(validation, setting.k_ref)
    knots = int(setting.fixed_knots or canonical_knots)
    alpha = float(canonical_alpha * setting.alpha_multiplier)
    spline, model = fit_model(
        train["soh"][train_keep], train_residual[train_keep],
        knots, setting.degree, alpha,
    )
    validation_prediction = predict(model, spline, validation["soh"][validation_keep])
    validation_error = np.sqrt(np.mean(
        (validation_prediction - validation_residual[validation_keep].reshape(validation_prediction.shape)) ** 2,
        axis=1,
    ))
    constant = train_residual[train_keep].mean(axis=0)
    validation_baseline = np.sqrt(np.mean(
        (validation_residual[validation_keep] - constant) ** 2, axis=(1, 2)
    ))
    selected_battery = battery_values(validation_error, validation["cells"][validation_keep])
    baseline_battery = battery_values(validation_baseline, validation["cells"][validation_keep])
    paired = pd.concat(
        [selected_battery.rename("selected"), baseline_battery.rename("baseline")], axis=1
    ).dropna()
    difference = (paired.selected - paired.baseline).to_numpy(dtype=np.float64)
    stable_offset = sum((index + 1) * ord(char) for index, char in enumerate(setting.setting_id))
    ci_low, ci_high = bootstrap_mean_difference(difference, stable_offset)
    selected_rmse = float(paired.selected.mean())
    baseline_rmse = float(paired.baseline.mean())
    improved_fraction = float((difference < 0).mean()) if len(difference) else float("nan")
    validation_r2 = float(1.0 - (
        battery_values(np.mean(
            (validation_prediction - validation_residual[validation_keep].reshape(validation_prediction.shape)) ** 2,
            axis=1,
        ), validation["cells"][validation_keep]).mean()
        / battery_values(np.mean(
            (validation_residual[validation_keep] - constant) ** 2, axis=(1, 2)
        ), validation["cells"][validation_keep]).mean()
    ))
    aggregate_improved = bool(selected_rmse < baseline_rmse and validation_r2 > 0)
    manifold_status = (
        "STRONG" if aggregate_improved and improved_fraction >= 0.70
        else "VALID" if aggregate_improved and improved_fraction > 0.50
        else "INVALID"
    )
    validation_info = {
        "manifold_status": manifold_status,
        "manifold_gate_passed": bool(
            selected_rmse / baseline_rmse < 0.95
            and improved_fraction > 0.50
            and validation_r2 > 0
            and ci_high < 0
        ),
        "manifold_RMSE": selected_rmse,
        "constant_baseline_RMSE": baseline_rmse,
        "manifold_to_constant_ratio": selected_rmse / baseline_rmse,
        "validation_battery_balanced_R2": validation_r2,
        "improved_batteries": int((difference < 0).sum()),
        "validation_batteries": int(len(difference)),
        "improved_fraction": improved_fraction,
        "selected_minus_constant_CI_low": ci_low,
        "selected_minus_constant_CI_high": ci_high,
    }
    train_prediction = predict(model, spline, train["soh"][train_keep])
    s_ind = float(np.sqrt(np.mean(
        (train_residual[train_keep].reshape(train_prediction.shape) - train_prediction) ** 2
    )))
    if not np.isfinite(s_ind) or s_ind <= 0:
        raise RuntimeError(f"{setting.setting_id}: invalid S_ind")

    test_residual, references, _ = residualize(test, setting.k_ref)
    eligible = canonical_eligible & np.isfinite(test_residual).all(axis=(1, 2))
    reference = np.stack([
        references[str(cell)] for cell in np.asarray(test["cells"])[eligible]
    ]).reshape(eligible.sum(), -1)
    clean_population = predict(model, spline, test["soh"][eligible])
    clean_individual = test_residual[eligible].reshape(clean_population.shape[0], -1) - clean_population
    clean_norm = np.sum(clean_individual.astype(np.float64) ** 2, axis=1)
    if np.any(clean_norm <= 0):
        raise RuntimeError(f"{setting.setting_id}: zero clean individual norm")
    projection = Projection(
        spline, model, float(np.min(train["soh"][train_keep])),
        float(np.max(train["soh"][train_keep])), setting.grid_size,
    )
    projection.audit(test_residual[eligible].reshape(eligible.sum(), -1))
    state = {
        "projection": projection,
        "reference": reference,
        "clean_individual": clean_individual.astype(np.float64),
        "clean_norm": clean_norm,
        "eligible": eligible,
        "s_ind": s_ind,
    }
    return state, validation_info


def canonical_battery_frame(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "battery_level_decomposition.csv")
    frame = frame.rename(columns={"cell": "battery_id", "condition": "condition_raw"})
    frame["corruption"] = frame["condition_raw"].map(DISPLAY_CORRUPTION)
    frame["severity"] = frame["severity"].astype(str)
    frame["severity_index"] = frame["severity_index"].astype(int)
    return frame[[
        "battery_id", "corruption", "severity", "severity_index",
        "E_pop", "E_ind", "Q", "C_ind", "soh_error", "n_cycles",
    ]].copy()


def frozen_manifold_selection(directory: Path) -> tuple[int, float]:
    """Read the already selected clean manifold instead of refitting candidates."""
    path = directory / "population_manifold_validation.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    selected = table[(table["row_type"] == "candidate") & table["selected"].astype(bool)]
    if len(selected) != 1:
        raise RuntimeError(f"{path}: expected one frozen selected manifold row")
    row = selected.iloc[0]
    return int(row["n_knots"]), float(row["alpha"])


def battery_from_cycle(frame: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    battery = frame.groupby("battery_id", as_index=False, sort=True).agg(
        E_pop=("E_pop", "median"), E_ind=("E_ind", "median"),
        Q=("Q", "median"), C_ind=("C_ind", "median"),
        n_cycles=("cycle_id", "size"),
    )
    lookup = canonical[[
        "battery_id", "corruption", "severity", "severity_index", "soh_error",
    ]]
    # The caller adds condition labels before this function is used.
    return battery.merge(lookup, on="battery_id", how="left")


def process_system(
    dataset: str,
    run,
    device: torch.device,
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    label, seed = C.SYSTEMS[dataset]
    certified = C.module()
    if smoke:
        raise RuntimeError("smoke systems are handled by smoke()")
    train = certified.load_clean(dataset, "train")
    validation = certified.load_clean(dataset, "vali")
    test = load_test_with_current(dataset)
    formal_dir = C.population_individual_run(dataset, seed)
    canonical = canonical_battery_frame(formal_dir)
    if len(canonical) == 0 or canonical.corruption.nunique() != 5:
        raise RuntimeError(f"{dataset}: incomplete formal canonical battery output")

    _, _, canonical_eligible = residualize(test, 5)
    canonical_knots, canonical_alpha = frozen_manifold_selection(formal_dir)

    states, validation_rows = {}, []
    for setting in SETTINGS:
        print(f"[{dataset}] fitting {setting.setting_id}", flush=True)
        state, validation_info = prepare_state(
            setting, train, validation, test,
            canonical_knots, canonical_alpha, canonical_eligible,
        )
        states[setting.setting_id] = state
        validation_rows.append({
            "dataset": label,
            **setting_row(setting, canonical_knots, canonical_alpha),
            **validation_info,
        })

    battery_rows = []
    condition_rows = []
    canonical_with_setting = canonical.copy()
    canonical_with_setting.insert(1, "setting_id", "canonical")
    canonical_with_setting.insert(2, "dataset", label)
    canonical_with_setting = canonical_with_setting.assign(
        parameter="canonical", value="canonical", is_canonical=True,
        K_ref=5, spline_degree=3, n_knots=canonical_knots,
        ridge_alpha=canonical_alpha, projection_grid=500,
    )
    battery_rows.append(canonical_with_setting)

    model, model_args, checkpoint = build_reconstruction_model(dataset, seed, device)
    del model_args
    for condition in DISPLAY_CORRUPTION:
        protocol_condition = condition
        for level_index in range(5):
            reconstructed = generate_stress_reconstruction(
                model, test, protocol_condition, level_index, device,
            )
            for setting in SETTINGS[1:]:
                state = states[setting.setting_id]
                cycle = reconstruction_metrics(reconstructed, test, state)
                cycle.insert(1, "corruption", DISPLAY_CORRUPTION[condition])
                cycle.insert(2, "severity", f"L{level_index + 1}")
                cycle.insert(3, "severity_index", level_index + 1)
                cycle.insert(4, "dataset", label)
                cycle.insert(5, "setting_id", setting.setting_id)
                battery = cycle.groupby("battery_id", as_index=False, sort=True).agg(
                    E_pop=("E_pop", "median"), E_ind=("E_ind", "median"),
                    Q=("Q", "median"), C_ind=("C_ind", "median"),
                    n_cycles=("cycle_id", "size"),
                )
                battery.insert(1, "corruption", DISPLAY_CORRUPTION[condition])
                battery.insert(2, "severity", f"L{level_index + 1}")
                battery.insert(3, "severity_index", level_index + 1)
                battery.insert(4, "dataset", label)
                battery.insert(5, "setting_id", setting.setting_id)
                battery = battery.merge(
                    canonical[["battery_id", "corruption", "severity", "soh_error"]],
                    on=["battery_id", "corruption", "severity"], how="left",
                    validate="many_to_one",
                )
                if battery.soh_error.isna().any():
                    raise RuntimeError(f"{dataset}/{condition}/L{level_index + 1}: SOH alignment failed")
                battery_rows.append(battery)
                condition_rows.extend(aggregate_condition(
                    battery,
                    {"dataset": label, **setting_row(setting, canonical_knots, canonical_alpha)},
                ))
            del reconstructed
            gc.collect()
            print(f"[{dataset}] {condition}/L{level_index + 1} complete", flush=True)

    # Fixed-knot settings equal to the selected canonical knot count by design.
    all_battery = pd.concat(battery_rows, ignore_index=True)
    equivalent = {
        setting.setting_id for setting in SETTINGS[1:]
        if setting.fixed_knots == canonical_knots
    }
    for setting_id in equivalent:
        all_battery = pd.concat([
            all_battery[all_battery.setting_id != setting_id],
            canonical_with_setting.assign(setting_id=setting_id, is_canonical=False),
        ], ignore_index=True)
    # Rebuild condition summaries after equivalent-setting binding.
    condition_rows = []
    for setting in SETTINGS:
        subset = all_battery[all_battery.setting_id == setting.setting_id]
        condition_rows.extend(aggregate_condition(
            subset, {"dataset": label, **setting_row(setting, canonical_knots, canonical_alpha)},
        ))
    conditions = pd.DataFrame(condition_rows)
    canonical_condition = conditions[conditions.setting_id == "canonical"][[
        "dataset", "corruption", "severity", "severity_index",
        "median_E_pop", "median_E_ind",
    ]].rename(columns={
        "median_E_pop": "canonical_median_E_pop",
        "median_E_ind": "canonical_median_E_ind",
    })
    conditions = conditions.merge(
        canonical_condition,
        on=["dataset", "corruption", "severity", "severity_index"],
        how="left", validate="many_to_one",
    )
    validation = pd.DataFrame(validation_rows)
    checkpoint_record = {dataset: str(checkpoint)}
    del model, train, test, states, canonical, canonical_with_setting
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return conditions, all_battery, validation, checkpoint_record, {"dataset": label, "seed": seed}


def build_system_summary(
    conditions: pd.DataFrame,
    battery: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_system_rows = []
    for (dataset, setting_id), group in validation.groupby(["dataset", "setting_id"], sort=False):
        condition = conditions[(conditions.dataset == dataset) & (conditions.setting_id == setting_id)]
        canonical = conditions[(conditions.dataset == dataset) & (conditions.setting_id == "canonical")]
        bgroup = battery[(battery.dataset == dataset) & (battery.setting_id == setting_id)]
        row = group.iloc[0].to_dict()
        row.update({
            "rho_condition_E_pop_vs_canonical": safe_spearman(
                condition.median_E_pop, canonical.median_E_pop
            ),
            "rho_condition_E_ind_vs_canonical": safe_spearman(
                condition.median_E_ind, canonical.median_E_ind
            ),
            "rho_E_ind_SOH": safe_spearman(bgroup.E_ind, bgroup.soh_error),
            "rho_E_ind_SOH_positive": bool(safe_spearman(bgroup.E_ind, bgroup.soh_error) > 0),
            "n_batteries": int(bgroup.battery_id.nunique()),
            "n_battery_conditions": int(len(bgroup)),
            "recoverability_statistics_included": True,
            "data_status": "READY",
        })
        by_system_rows.append(row)
    by_system = pd.DataFrame(by_system_rows)
    summary_rows = []
    for setting in SETTINGS:
        condition = conditions[conditions.setting_id == setting.setting_id]
        canonical = conditions[conditions.setting_id == "canonical"]
        group = by_system[by_system.setting_id == setting.setting_id]
        summary_rows.append({
            "setting_id": setting.setting_id,
            "parameter": setting.parameter,
            "value": setting.value,
            "is_canonical": setting.canonical,
            "K_ref": setting.k_ref,
            "spline_degree": setting.degree,
            "fixed_knots": setting.fixed_knots,
            "alpha_multiplier": setting.alpha_multiplier,
            "projection_grid": setting.grid_size,
            "rho_condition_E_pop_vs_canonical": safe_spearman(
                condition.median_E_pop, canonical.median_E_pop
            ),
            "rho_condition_E_ind_vs_canonical": safe_spearman(
                condition.median_E_ind, canonical.median_E_ind
            ),
            "median_system_landscape_rho_E_pop": float(group.rho_condition_E_pop_vs_canonical.median()),
            "median_system_landscape_rho_E_ind": float(group.rho_condition_E_ind_vs_canonical.median()),
            "median_rho_E_ind_SOH": float(group.rho_E_ind_SOH.median()),
            "positive_rho_E_ind_SOH_systems": int((group.rho_E_ind_SOH > 0).sum()),
            "recoverability_systems": int(len(group)),
            "valid_manifold_systems": int(group.manifold_status.isin(["VALID", "STRONG"]).sum()),
            "manifold_gate_pass_systems": int(group.manifold_gate_passed.eq(True).sum()),
        })
    return pd.DataFrame(summary_rows), by_system


def smoke(run_id: str | None = None) -> Path:
    run = canonical_run("r3_stress_sensitivity", run_id or "debug_smoke")
    rng = np.random.default_rng(11)
    cells, cycles, voltage, soh = [], [], [], []
    for cell in ("smoke1", "smoke2"):
        for cycle in range(1, 13):
            cells.append(cell); cycles.append(cycle)
            voltage.append((3.2 - 0.01 * cycle + rng.normal(0, 1e-3, 60))[:, None])
            soh.append(1.0 - 0.01 * cycle)
    data = {
        "voltage": np.stack(voltage).astype(np.float32),
        "soh": np.asarray(soh, dtype=np.float32),
        "cells": np.asarray(cells), "cycles": np.asarray(cycles),
    }
    residual, _, keep = residualize(data, 5)
    spline, model = fit_model(data["soh"][keep], residual[keep], 4, 3, 0.1)
    projection = Projection(spline, model, 0.85, 1.0, 20)
    projection.audit(residual[keep].reshape(14, -1))
    frame = pd.DataFrame({
        "dataset": ["SMOKE"], "setting_id": ["canonical"],
        "n_settings": [len(SETTINGS)], "n_corruptions": [5],
        "n_severities": [5], "projection_audit": ["PASS"],
        "new_model_training": [0], "new_model_inference": [0],
    })
    run.write_table("analysis", "sensitivity_smoke.csv", frame)
    run.write_json("sensitivity_config.json", {
        "status": "SMOKE_PASS", "settings": [s.setting_id for s in SETTINGS],
        "projection_audit": "PASS",
    })
    run.write_manifest(
        config={"status": "SMOKE_PASS", "new_model_training": 0, "new_model_inference": 0},
        input_paths={"synthetic": "generated_in_memory"}, seeds=[11],
    )
    print(f"R3 stress sensitivity smoke outputs -> {run.root}")
    return run.root


def run(
    run_id: str | None = None,
    datasets: list[str] | None = None,
    smoke_flag: bool = False,
    device_name: str = "cuda",
) -> Path:
    if smoke_flag:
        return smoke(run_id)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen stress reconstruction")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    selected = datasets or list(C.SYSTEMS)
    unknown = sorted(set(selected) - set(C.SYSTEMS))
    if unknown:
        raise ValueError(f"unknown systems: {unknown}")
    run = canonical_run("r3_stress_sensitivity", run_id)
    all_conditions, all_battery, all_validation = [], [], []
    checkpoints, input_paths, seeds = {}, {}, []
    for dataset in selected:
        print(f"[{dataset}] loading clean splits and frozen checkpoint", flush=True)
        conditions, battery, validation, checkpoint, metadata = process_system(
            dataset, run, device,
        )
        all_conditions.append(conditions)
        all_battery.append(battery)
        all_validation.append(validation)
        checkpoints.update(checkpoint)
        input_paths[dataset] = str(reconstruction_result_directory(dataset, metadata["seed"]))
        seeds.append(metadata["seed"])
        print(f"[{dataset}] complete", flush=True)
    conditions = pd.concat(all_conditions, ignore_index=True)
    battery = pd.concat(all_battery, ignore_index=True)
    validation = pd.concat(all_validation, ignore_index=True)
    summary, by_system = build_system_summary(conditions, battery, validation)

    run.write_table("raw", "sensitivity_battery_level.csv.gz", battery)
    run.write_table("analysis", "sensitivity_condition_level.csv", conditions)
    run.write_table("analysis", "sensitivity_by_system.csv", by_system)
    run.write_table("analysis", "sensitivity_summary.csv", summary)
    run.write_table("provenance", "sensitivity_validation.csv", validation)
    config = {
        "systems": selected,
        "settings": [setting_row(s, 0, 0.0) for s in SETTINGS],
        "corruptions": list(DISPLAY_CORRUPTION.values()),
        "severities": [f"L{i}" for i in range(1, 6)],
        "aggregation": "cycle -> battery-condition median -> system-condition median",
        "manifold_fit": "clean train only; canonical knot/alpha selected on clean validation",
        "stress_protocol": "source_code/utils/stress_corruptions.py",
        "frozen_reconstruction": "source_code/checkpoints/Reconstruction/*/checkpoint.pth",
        "new_model_training": 0,
        "new_model_inference": 1,
        "new_corruption_generation": 0,
        "inference_note": "Frozen BatteryDiT was evaluated with the registered deterministic stress protocol; no weights were changed.",
    }
    run.write_json("sensitivity_config.json", config)
    run.write_manifest(
        config=config, input_paths=input_paths, checkpoint_ids=checkpoints,
        seeds=seeds,
    )
    print(f"R3 stress sensitivity outputs -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    command.add_argument("--datasets", nargs="*", choices=tuple(C.SYSTEMS), default=None)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    command.add_argument("--smoke", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.run_id, args.datasets, args.smoke, args.device)


if __name__ == "__main__":
    main()
