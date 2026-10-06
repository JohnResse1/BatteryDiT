"""R4-03/04/05 canonical sliding-occlusion workflow schema and statistics.

This phase validates schema and the migrated position-controlled diagnostic
without running full occlusion inference. Formal inference remains a later
Node1 run. The exact old diagnostic definition is reproduced below.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from . import _certified as C
from .provenance import canonical_run

WINDOW_WIDTHS = (0.10, 0.20, 0.30)
MAIN_WIDTH = 0.20
WINDOW_STRIDE = 0.05


def window_definitions(length: int, width_fraction: float = MAIN_WIDTH,
                       stride_fraction: float = WINDOW_STRIDE) -> list[dict]:
    start = 0.0
    definitions = []
    while round(start + width_fraction, 6) <= 1.0 + 1e-12:
        begin = int(round(start * length))
        end = int(round((start + width_fraction) * length))
        definitions.append({
            "window_id": f"w{start:.2f}",
            "window_start": float(start),
            "window_end": float(start + width_fraction),
            "window_center": float(start + width_fraction / 2),
            "start_index": begin,
            "end_index_exclusive": end,
            "window_width": float(width_fraction),
        })
        start = round(start + stride_fraction, 6)
    return definitions


def raw_schema(length: int = 1000, width_fraction: float = MAIN_WIDTH) -> pd.DataFrame:
    rows = []
    for definition in window_definitions(length, width_fraction):
        rows.append({
            "battery_id": "<battery>",
            "cycle_id": -1,
            "window_start": definition["window_start"],
            "window_end": definition["window_end"],
            "window_center": definition["window_center"],
            "window_width": definition["window_width"],
            "waveform_damage": np.nan,
            "S_pop": np.nan,
            "S_ind": np.nan,
            "E_pop_clean": np.nan,
            "E_pop_occluded": np.nan,
            "E_ind_clean": np.nan,
            "E_ind_occluded": np.nan,
            "C_ind_clean": np.nan,
            "C_ind_occluded": np.nan,
            "Delta_E_pop": np.nan,
            "Delta_E_ind": np.nan,
            "Delta_C_ind": np.nan,
        })
    return pd.DataFrame(rows)


def position_controlled_diagnostic(profile: pd.DataFrame) -> pd.DataFrame:
    """Exact old-rank-partial migration.

    raw Spearman; partial Spearman obtained by rank-transforming x/y/window_start,
    regressing each rank on [1, rank(window_start)] with OLS and correlating the
    residuals; leave-one-window-out min/max; correlation excluding window_start=0;
    and support/damage correlations with window position.
    """
    def rank_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
        design = np.column_stack([np.ones_like(rz), rz])
        def residual(values):
            return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        return float(spearmanr(residual(rx), residual(ry)).statistic)

    relationships = (
        ("S_pop", "Delta_E_pop", "S_pop_vs_Delta_E_pop"),
        ("S_ind", "Delta_E_ind", "S_ind_vs_Delta_E_ind"),
        ("S_ind", "C_ind_loss", "S_ind_vs_C_ind_loss"),
        ("waveform_damage", "Delta_E_ind", "waveform_damage_vs_Delta_E_ind"),
        ("waveform_damage", "C_ind_loss", "waveform_damage_vs_C_ind_loss"),
        ("waveform_damage", "Delta_E_pop", "waveform_damage_vs_Delta_E_pop"),
    )
    rows = []
    for x, y, label in relationships:
        if x not in profile or y not in profile or len(profile) < 4:
            continue
        raw = float(spearmanr(profile[x], profile[y]).statistic)
        partial = rank_partial(profile[x].to_numpy(float), profile[y].to_numpy(float),
                               profile["window_start"].to_numpy(float))
        leave_one_out = [float(spearmanr(profile.drop(index)[x], profile.drop(index)[y]).statistic)
                         for index in profile.index]
        excluded_zero = profile[profile.window_start > 0]
        rows.append({
            "relationship": label,
            "spearman_rho": raw,
            "partial_spearman_controlling_window_start": partial,
            "leave_one_window_out_min": min(leave_one_out),
            "leave_one_window_out_max": max(leave_one_out),
            "spearman_excluding_window_at_zero": float(spearmanr(
                excluded_zero[x], excluded_zero[y]).statistic),
            "spearman_support_vs_position": float(spearmanr(
                profile.window_start, profile[x]).statistic),
            "spearman_damage_vs_position": float(spearmanr(
                profile.window_start, profile[y]).statistic),
        })
    return pd.DataFrame(rows)


def smoke(run_id: str | None = None) -> Path:
    run = canonical_run("r4_sliding_occlusion_schema", run_id)
    schema = raw_schema()
    run.write_table("raw", "sliding_occlusion_raw_schema.csv", schema)
    synthetic = schema.copy()
    synthetic["S_pop"] = np.linspace(1, 0, len(synthetic))
    synthetic["S_ind"] = np.linspace(0, 1, len(synthetic))
    synthetic["Delta_E_ind"] = np.linspace(0, 1, len(synthetic))
    synthetic["C_ind_loss"] = np.linspace(0, 1, len(synthetic))
    synthetic["Delta_E_pop"] = np.linspace(0.2, 0, len(synthetic))
    synthetic["waveform_damage"] = np.linspace(0, 1, len(synthetic))
    diagnostic = position_controlled_diagnostic(synthetic)
    run.write_table("analysis", "position_controlled_diagnostic_smoke.csv", diagnostic)
    config = {
        "widths": list(WINDOW_WIDTHS),
        "main_width": MAIN_WIDTH,
        "stride": WINDOW_STRIDE,
        "formal_associations": [
            "S_ind vs Delta_E_ind", "S_ind vs C_ind_loss", "S_pop vs Delta_E_pop",
            "waveform_damage vs Delta_E_ind", "waveform_damage vs C_ind_loss",
            "waveform_damage vs Delta_E_pop"],
        "status": "READY_FOR_FORMAL_RUN",
    }
    run.write_json("sliding_occlusion_config.json", config)
    run.write_manifest(config=config, input_paths={
        "clean_test_inputs": "source_code/data/mit/test/clean",
        "spatial_support_workflow": "analysis/workflows/spatial_support.py"})
    print(f"R4 occlusion schema smoke -> {run.root}")
    return run.root


def _window_bootstrap(x: np.ndarray, y: np.ndarray, seed: int = 20260926,
                      n: int = 2000) -> tuple[float, float, float]:
    """Whole-window bootstrap for the position-level association."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n):
        index = rng.integers(0, len(x), size=len(x))
        value = float(spearmanr(x[index], y[index]).statistic)
        if np.isfinite(value):
            estimates.append(value)
    low, high = (np.percentile(estimates, (2.5, 97.5))
                 if estimates else (np.nan, np.nan))
    return observed, float(low), float(high)


def _profile(metrics: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    from .spatial_support import window_definitions

    definitions = window_definitions(len(support))
    rows = []
    for width in WINDOW_WIDTHS:
        sub = metrics[np.isclose(metrics.width_fraction, width)]
        if sub.empty:
            continue
        for window_id, group in sub.groupby("window_id", sort=True):
            definition = next((item for item in definitions
                               if item["window_id"] == window_id), None)
            if definition is None:
                continue
            start, end = definition["start_index"], definition["end_index_exclusive"]
            rows.append({
                "width_fraction": float(width), "window_id": window_id,
                "window_start": float(group.window_start.iloc[0]),
                "waveform_damage": float(group.waveform_damage.median()),
                "Delta_E_pop": float(group.Delta_E_pop.median()),
                "Delta_E_ind": float(group.Delta_E_ind.median()),
                "C_ind_loss": float(group.C_ind_loss.median()),
                "S_pop": float(support.P_pop.iloc[start:end].sum()),
                "S_ind": float(support.P_ind.iloc[start:end].sum()),
                "n_cycles": int(len(group)),
            })
    return pd.DataFrame(rows)


def _association(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    relationships = (
        ("S_pop", "Delta_E_pop", "S_pop_vs_Delta_E_pop"),
        ("S_ind", "Delta_E_ind", "S_ind_vs_Delta_E_ind"),
        ("S_ind", "C_ind_loss", "S_ind_vs_C_ind_loss"),
        ("waveform_damage", "Delta_E_ind", "waveform_damage_vs_Delta_E_ind"),
        ("waveform_damage", "C_ind_loss", "waveform_damage_vs_C_ind_loss"),
        ("waveform_damage", "Delta_E_pop", "waveform_damage_vs_Delta_E_pop"),
    )
    for width, group in profile.groupby("width_fraction", sort=True):
        for x, y, label in relationships:
            if len(group) < 4:
                continue
            rho, low, high = _window_bootstrap(group[x], group[y])
            rows.append({"width_fraction": float(width), "relationship": label,
                         "x": x, "y": y, "spearman_rho": rho,
                         "bootstrap_low": low, "bootstrap_high": high,
                         "n_windows": int(len(group))})
        diagnostic = position_controlled_diagnostic(group)
        if not diagnostic.empty:
            diagnostic.insert(0, "width_fraction", float(width))
            rows.extend(diagnostic.to_dict("records"))
    return pd.DataFrame(rows)


def _synthetic_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for definition in window_definitions(1000, MAIN_WIDTH):
        for cell in ("smoke1", "smoke2", "smoke3", "smoke4"):
            rows.append({"window_id": definition["window_id"], "model_seed": 2021,
                         "cell": cell, "cycle": 10,
                         "width_fraction": definition["window_width"],
                         "window_start": definition["window_start"],
                         "waveform_damage": definition["window_start"] + 0.1,
                         "E_pop_clean": 0.01, "E_pop_occluded": 0.02 + definition["window_start"],
                         "E_ind_clean": 0.0, "E_ind_occluded": 0.2 + definition["window_start"],
                         "Q_clean": 1.0, "Q_occluded": 0.9,
                         "C_ind_clean": 1.0, "C_ind_occluded": 0.8,
                         "Delta_E_pop": 0.01 + definition["window_start"],
                         "Delta_E_ind": 0.2 + definition["window_start"],
                         "C_ind_loss": 0.2})
    raw = pd.DataFrame(rows)
    support = pd.DataFrame({"position": np.arange(1000),
                            "P_pop": np.ones(1000), "P_ind": np.ones(1000)})
    return raw, support


def run(run_id: str | None = None, seeds=(2021, 2022, 2023),
        device: str | None = None, smoke: bool = False) -> Path:
    """Run the formal R4-03/R4-04/R4-05 occlusion analysis."""
    run = canonical_run("r4_sliding_occlusion", run_id)
    if smoke:
        metrics, support = _synthetic_metrics()
        input_paths = {"synthetic": "smoke"}
    else:
        from .sliding_occlusion_runner import run as run_inference
        from .spatial_support import clean_spatial_support

        _, metrics = run_inference(seeds=tuple(seeds), device=device)
        support = clean_spatial_support(C.load_sequences("mit", ("test",)))
        input_paths = {
            "clean_data": "source_code/data/mit/test/clean",
            "frozen_checkpoints": "source_code/checkpoints/Reconstruction/mit",
        }
    if metrics.empty:
        raise RuntimeError("sliding occlusion produced no formal metric rows")
    profile = _profile(metrics, support)
    associations = _association(profile)
    run.write_table("raw", "sliding_occlusion_metrics_raw.csv.gz", metrics)
    run.write_table("raw", "spatial_support_raw.csv", support)
    run.write_table("analysis", "sliding_occlusion_window_profile.csv", profile)
    run.write_table("analysis", "sliding_occlusion_associations.csv", associations)
    run.write_table("plotting", "sliding_occlusion_plotting.csv", associations)
    config = {"dataset": "mit", "widths": list(WINDOW_WIDTHS),
              "stride": WINDOW_STRIDE, "seeds": list(seeds),
              "main_width": MAIN_WIDTH, "status": "SMOKE" if smoke else "COMPLETE",
              "associations": ["S_ind vs Delta_E_ind", "S_ind vs C_ind_loss",
                               "S_pop vs Delta_E_pop", "waveform_damage vs Delta_E_ind",
                               "waveform_damage vs C_ind_loss",
                               "waveform_damage vs Delta_E_pop"]}
    run.write_json("sliding_occlusion_config.json", config)
    run.write_manifest(config=config, input_paths=input_paths, seeds=list(seeds))
    print(f"R4 sliding occlusion outputs -> {run.root}")
    return run.root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id")
    command.add_argument("--smoke", action="store_true")
    command.add_argument("--device")
    command.add_argument("--seeds", nargs="*", type=int, default=[2021, 2022, 2023])
    return command


def main() -> None:
    args = parser().parse_args()
    run(args.run_id, tuple(args.seeds), args.device, args.smoke)


if __name__ == "__main__":
    main()
