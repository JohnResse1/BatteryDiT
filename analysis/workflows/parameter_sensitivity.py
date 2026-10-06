"""R3-05 one-factor-at-a-time sensitivity, migrated from the old Table 11.

The old grid is preserved exactly:
    canonical, K_ref 3/7, spline degree 2, fixed knots 4/5/6,
    ridge alpha 0.1x/10x, projection grid 200/1000.

The only semantic correction is the current canonical Individual residual
    R = (V - V_ref) - M(SOH)
and train-only manifold fitting with validation-only selection. No new variant
and no downstream inference is introduced. Systems without a complete formal
PopulationIndividual result are skipped, never imputed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.utils.paths import RESULTS_ROOT
from . import _certified as C
from .provenance import canonical_run


@dataclass(frozen=True)
class Setting:
    setting_id: str
    parameter: str
    value: str
    k_ref: int | None = None
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


def _residualize_canonical(data, k_ref: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Apply the old K_ref setting while enforcing the canonical residual.

    K_ref is an early-life reference cycle-count variant. It does not change
    the canonical Individual residual; M(SOH) is still subtracted after the
    reference deviation is formed.
    """
    voltage = np.asarray(data["voltage"], dtype=np.float32)
    cells = np.asarray(data["cells"]).astype(str)
    cycles = np.asarray(data["cycles"], dtype=int)
    deviation = np.full_like(voltage, np.nan)
    references: dict[str, np.ndarray] = {}
    keep = np.zeros(len(cells), dtype=bool)
    for cell in sorted(set(cells)):
        indices = np.where(cells == cell)[0]
        indices = indices[np.argsort(cycles[indices], kind="stable")]
        if len(indices) < k_ref:
            continue
        reference_indices = indices[:k_ref]
        reference = np.median(voltage[reference_indices], axis=0)
        references[cell] = reference
        deviation[indices] = voltage[indices] - reference
        keep[indices[k_ref:]] = True
    return deviation, references, keep


def _fit(setting: Setting, train, vali, knots: int, alpha: float):
    certified = C.module()
    train_dev, _, train_keep = _residualize_canonical(train, setting.k_ref or 5)
    vali_dev, _, vali_keep = _residualize_canonical(vali, setting.k_ref or 5)
    # Fit the actual candidate model on canonical train deviation (train-only).
    spline, model = certified.fit_model(
        train["soh"][train_keep], train_dev[train_keep], knots, alpha)
    prediction = certified.predict(model, spline, vali["soh"][vali_keep])
    errors = np.sqrt(np.mean((prediction - vali_dev[vali_keep]) ** 2, axis=(1, 2)))
    by_battery = certified.battery_values(errors, vali["cells"][vali_keep])
    constant = np.mean(train_dev[train_keep], axis=0)
    baseline = np.sqrt(np.mean((vali_dev[vali_keep] - constant) ** 2, axis=(1, 2)))
    baseline_battery = certified.battery_values(baseline, vali["cells"][vali_keep])
    return spline, model, by_battery, baseline_battery, train_dev, vali_dev, train_keep, vali_keep


def _nearest(deviation, spline, model, train_soh, grid_size):
    h = np.linspace(float(np.min(train_soh)), float(np.max(train_soh)), int(grid_size), dtype=np.float32)
    population = C.module().predict(model, spline, h)
    selected = np.empty(len(deviation), dtype=np.int32)
    for start in range(0, len(deviation), 256):
        block = deviation[start:start + 256]
        selected[start:start + len(block)] = np.argmin(
            np.sqrt(np.mean((block[:, None, :, 0] - population[None, :, :, 0]) ** 2, axis=2)), axis=1)
    return h[selected], population[selected]


def _decompose(setting, train, vali, test, knots, alpha):
    certified = C.module()
    spline, model, selected_battery, baseline_battery, train_dev, vali_dev, train_keep, vali_keep = _fit(
        setting, train, vali, knots, alpha)
    test_dev, references, test_keep = _residualize_canonical(test, setting.k_ref or 5)
    test_dev = test_dev[test_keep]
    cells = np.asarray(test["cells"]).astype(str)[test_keep]
    cycles = np.asarray(test["cycles"], dtype=int)[test_keep]
    soh = np.asarray(test["soh"], dtype=float)[test_keep]
    h_rec, clean_population = _nearest(test_dev, spline, model, train["soh"], setting.grid_size)
    clean_individual = test_dev - clean_population
    clean_norm = np.sum(clean_individual ** 2, axis=(1, 2))
    reference = np.stack([references[str(cell)] for cell in cells])
    clean_frame = pd.DataFrame({
        "battery_id": cells, "cycle_id": cycles, "true_soh": soh,
        "E_pop": np.abs(h_rec - soh),
        "E_ind": np.sqrt(np.mean(clean_individual ** 2, axis=(1, 2))) / float(np.sqrt(np.mean((train_dev[train_keep] - certified.predict(model, spline, train["soh"][train_keep])) ** 2))),
        "n_cycles": 1,
    })
    return clean_frame


def _synthetic_block(n_cells=2, n_cycles=12, seed=7):
    rng = np.random.default_rng(seed)
    cells, cycles, voltage, soh, keys = [], [], [], [], []
    for cell_index in range(n_cells):
        cell=f"smoke{cell_index+1}"
        for cycle in range(1,n_cycles+1):
            base=3.2-0.1*cycle/n_cycles
            voltage.append((base+rng.normal(0,1e-4,(60,1))).astype(np.float32))
            soh.append(float(1.0-0.01*cycle)); cells.append(cell); cycles.append(cycle); keys.append(f"{cell}/{cycle}.npz")
    return {"voltage":np.stack(voltage),"soh":np.asarray(soh,dtype=np.float32),"keys":np.asarray(keys),"cells":np.asarray(cells),"cycles":np.asarray(cycles,dtype=int)}


def run(run_id: str | None = None, datasets: list[str] | None = None, smoke: bool = False) -> Path:
    if smoke:
        C.module().load_clean = lambda dataset, split: _synthetic_block(2, 12, seed=hash(split)%1000)
        C.population_individual_run = lambda dataset, seed: Path("/tmp/smoke_population")
        Path("/tmp/smoke_population").mkdir(exist_ok=True)
        (Path("/tmp/smoke_population")/"summary.json").write_text('{"status":"COMPLETE"}')
        systems = ["smoke"]
        C.SYSTEMS = {"smoke": ("SMOKE", 1)}
    else:
        systems = datasets or list(C.SYSTEMS)
    run = canonical_run("r3_sensitivity", run_id)
    config = {
        "settings": [vars(s) for s in SETTINGS],
        "systems_requested": systems,
        "individual_residual": "R=(V-V_ref)-M(SOH)",
        "train_only_fit": True,
        "validation_only_selection": True,
        "note": "missing systems are skipped, not imputed",
    }
    condition_rows, by_system_rows, summary_rows = [], [], []
    input_paths = {}
    for dataset in systems:
        label, seed = C.SYSTEMS[dataset]
        directory = C.population_individual_run(dataset, seed)
        if not (directory / "summary.json").is_file():
            by_system_rows.append({
                "dataset": label, "status": "SKIPPED_NO_FORMAL_RESULT"})
            continue
        input_paths[dataset] = str(directory)
        train = C.module().load_clean(dataset, "train")
        vali = C.module().load_clean(dataset, "vali")
        test = C.module().load_clean(dataset, "test")
        canonical = C.module().select_manifold(train, vali)
        canonical_knots, canonical_alpha = canonical[4]["n_knots"], canonical[4]["alpha"]
        for setting in SETTINGS:
            knots = int(setting.fixed_knots or canonical_knots)
            alpha = float(canonical_alpha * setting.alpha_multiplier)
            frame = _decompose(setting, train, vali, test, knots, alpha)
            frame["dataset"] = label
            frame["setting_id"] = setting.setting_id
            frame["parameter"] = setting.parameter
            frame["value"] = setting.value
            condition_rows.append(frame)
            # condition summary matching old aggregation: battery-condition median
            battery = frame.groupby(["battery_id"], as_index=False).agg(
                median_E_pop=("E_pop", "median"), median_E_ind=("E_ind", "median"),
                n_cycles=("cycle_id", "size"))
            battery["dataset"] = label
            battery["setting_id"] = setting.setting_id
            battery["parameter"] = setting.parameter
            battery["value"] = setting.value
            by_system_rows.append(battery)
    conditions = pd.concat(condition_rows, ignore_index=True)
    by_system = pd.concat(by_system_rows, ignore_index=True)
    run.write_table("raw", "sensitivity_condition_level.csv.gz", conditions)
    run.write_table("analysis", "sensitivity_by_system.csv", by_system)

    summary = by_system.groupby(["dataset", "setting_id"], as_index=False).agg(
        median_E_pop=("median_E_pop", "median"),
        median_E_ind=("median_E_ind", "median"))
    canonical_summary = summary[summary.setting_id == "canonical"]
    for metric in ("median_E_pop", "median_E_ind"):
        mapping = canonical_summary.set_index("dataset")[metric]
        summary[f"rho_{metric}_vs_canonical"] = summary.apply(
            lambda row: float(spearmanr(
                by_system[(by_system.dataset == row.dataset) & (by_system.setting_id == row.setting_id)][metric],
                by_system[(by_system.dataset == row.dataset) & (by_system.setting_id == "canonical")][metric],
            ).statistic) if mapping.get(row.dataset) is not None else np.nan, axis=1)
    summary["canonical_compatible"] = summary.setting_id == "canonical"
    run.write_table("analysis", "sensitivity_summary.csv", summary)
    run.write_json("sensitivity_config.json", config)
    run.write_manifest(
        config=config,
        input_paths=input_paths,
        seeds=[C.SYSTEMS[d][1] for d in systems if d in C.SYSTEMS],
    )
    print(f"R3 sensitivity smoke/analysis outputs -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    command.add_argument("--datasets", nargs="*", default=None)
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.run_id, args.datasets)


if __name__ == "__main__":
    main()
