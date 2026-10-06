"""Export frozen population/individual decomposition provenance.

This workflow does not re-run inference. It reconstructs the canonical
decomposition from clean train/validation data, records the exact selected
configuration and the deterministic early-reference manifest, and serialises
the fitted spline/Ridge parameters so downstream analysis can reload them
without refitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.utils.paths import RESULTS_ROOT
from . import _certified as C
from .provenance import canonical_run


def _split_manifest(certified, dataset: str) -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    for split in ("train", "vali", "test"):
        block = certified.load_clean(dataset, split)
        cells = np.asarray(block["cells"]).astype(str)
        manifest[split] = {
            "n_samples": int(len(cells)),
            "n_batteries": int(pd_unique_count(cells)),
            "n_cycles": int(np.asarray(block["cycles"]).size),
        }
    return manifest


def pd_unique_count(values) -> int:
    return int(len(set(values)))


def _synthetic_clean(n_cells: int, n_cycles: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    cells, cycles, voltage, soh, keys = [], [], [], [], []
    for cell_index in range(n_cells):
        cell = f"smoke{cell_index + 1}"
        for cycle_index in range(1, n_cycles + 1):
            base = 3.2 - 0.1 * cycle_index / max(1, n_cycles)
            voltage.append((base + rng.normal(0, 1e-4, (100, 1))).astype(np.float32))
            soh.append(float(1.0 - 0.01 * cycle_index))
            cells.append(cell); cycles.append(cycle_index)
            keys.append(f"{cell}/{cycle_index}.npz")
    return {
        "voltage": np.stack(voltage), "soh": np.asarray(soh, dtype=np.float32),
        "keys": np.asarray(keys), "cells": np.asarray(cells), "cycles": np.asarray(cycles, dtype=int),
    }


def run(dataset: str, run_id: str | None = None, smoke: bool = False) -> Path:
    certified = C.module()
    if smoke:
        def fake_load_clean(_dataset, split):
            return _synthetic_clean(2, 10 if split != "test" else 8, seed=hash(split) % 1000)
        certified.load_clean = fake_load_clean
    train = certified.load_clean(dataset, "train")
    vali = certified.load_clean(dataset, "vali")
    test = certified.load_clean(dataset, "test")
    spline, model, s_ind, selection_table, manifold = certified.select_manifold(
        train, vali)

    run = canonical_run("population_individual_provenance", run_id)

    # deterministic early-reference manifest for all three clean splits
    reference_rows = []
    for split, block, require_five in (
        ("train", train, True), ("vali", vali, True), ("test", test, False)
    ):
        residual, references, keep = certified.residualize(
            block, require_five=require_five)
        cells = np.asarray(block["cells"]).astype(str)
        cycles = np.asarray(block["cycles"], dtype=int)
        eligible = set(zip(cells[np.asarray(keep, bool)],
                           cycles[np.asarray(keep, bool)]))
        for cell in sorted(set(cells)):
            indices = np.where(cells == cell)[0]
            indices = indices[np.argsort(cycles[indices], kind="stable")]
            reference = references.get(str(cell))
            reference_cycles = cycles[indices[:certified.REFERENCE_CYCLES]]
            if reference is None:
                excluded = "battery_has_fewer_than_five_cycles"
            else:
                excluded = int(sum(
                    (cell, int(cycle)) not in eligible
                    for cycle in cycles[indices]
                ))
            reference_rows.append({
                "dataset": dataset,
                "split": split,
                "battery_id": cell,
                "reference_cycle_min": int(reference_cycles.min()) if len(reference_cycles) else -1,
                "reference_cycle_max": int(reference_cycles.max()) if len(reference_cycles) else -1,
                "n_reference_cycles": int(len(reference_cycles)),
                "n_excluded_reference_forming_cycles": excluded,
                "reference_is_defined": reference is not None,
            })
    references_frame = pd.DataFrame(reference_rows)
    run.write_table("provenance", "early_reference_manifest.csv", references_frame)

    selection_table.to_csv(
        run.provenance / "population_manifold_selection.csv", index=False)

    # serialise the frozen parameters. SplineTransformer does not need pickling:
    # its knots are recoverable from fitted input range and configuration.
    knots = np.asarray([getattr(bs, "t", []) for bs in getattr(spline, "bsplines_", [])] if hasattr(spline, "bsplines_") and spline.bsplines_ else getattr(spline, "knots", []), dtype=object)
    coefficients = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.asarray(model.intercept_, dtype=np.float32)
    np.savez_compressed(
        run.provenance / "population_manifold_parameters.npz",
        spline_knots=knots,
        spline_n_knots=np.asarray(manifold["n_knots"], dtype=int),
        spline_degree=np.asarray(3, dtype=int),
        ridge_alpha=np.asarray(manifold["alpha"], dtype=float),
        ridge_coefficients=coefficients,
        ridge_intercept=intercept,
        train_soh_min=np.asarray(train["soh"].min(), dtype=float),
        train_soh_max=np.asarray(train["soh"].max(), dtype=float),
        S_ind=np.asarray(s_ind, dtype=float),
    )

    config = {
        "dataset": dataset,
        "individual_residual": "R=(V-V_ref)-M(SOH)",
        "population_definition": "M(SOH) from train-only SplineTransformer+Ridge",
        "early_reference": "median of earliest 5 clean cycles",
        "test_reference_policy": "train/val require 5; test drops cycles after 5 (new stricter policy)",
        "projection_grid": certified.GRID_SIZE,
        "manifold": manifold,
        "S_ind": s_ind,
    }
    (run.provenance / "population_manifold_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")
    analysis_config = {
        "schema_version": "canonical-r3-r5-v1",
        "dataset": dataset,
        "systems_allowed_for_pop_ind": list(C.SYSTEMS),
        "sdu_excluded_reason": "no compatible fixed nominal-capacity SOH target",
        "output_layers": ["raw", "analysis", "plotting", "provenance"],
    }
    (run.provenance / "analysis_config.json").write_text(
        json.dumps(analysis_config, indent=2, sort_keys=True) + "\n")

    run.write_manifest(
        config=config,
        input_paths={
            "population_result": str(
                RESULTS_ROOT / "PopulationIndividual" / dataset),
            "clean_data": f"source_code/data/{dataset}/{{train,vali,test}}/clean",
        },
        seeds=[int(manifold.get("model_seed", -1))],
        split_manifest=json.dumps(_split_manifest(certified, dataset), sort_keys=True),
    )
    print(f"provenance export -> {run.root}")
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
