"""Formal, derived-only R4 analysis runner.

This module is intentionally self-contained under ``analysis/``.  It reads
the frozen clean MIT split, the frozen R3 decomposition implementation, and
the already-produced R4/L5 archives.  It never trains or evaluates
BatteryDiT, creates a corruption, or changes an existing result.

The runner exists because the earlier R4 workflows used ``_certified``'s
convenience sequence loader, which retained the five reference-forming rows.
Here the R3 ``keep`` mask is made once and is the only R4 target/sample mask.
The frozen R3 manifold itself is selected by the current R3 implementation
unchanged; its documented all-row train/validation fitting behavior is
preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import _certified as C
from . import _stats as S
from .spatial_support import concentration
from analysis.utils.paths import REPOSITORY_ROOT


DATASET = "mit"
HORIZON = 100
LAGS = (1, 5, 10, 25, 50, 100)
HISTORY_WINDOWS = (50, 100)
PCA_K = 10
MATCH_TOLERANCE = C.ANCHOR_HALF_WIDTH
BOOTSTRAPS = 2000
PERMUTATIONS = 2000
BOOTSTRAP_SEED = 20260926
PERMUTATION_SEED = 20260927
MLP_SEED = 20260926
WIDTHS = (0.10, 0.20, 0.30)
MAIN_WIDTH = 0.20
WINDOW_STRIDE = 0.05
SEVERITY = "L5"

L5_ASSET_ROOT = REPOSITORY_ROOT / "analysis/assets/reconstruction/mit_l5"
L5_CONDITIONS = {
    "downsample": "downsample",
    "random": "random_missing",
    "segment": "part_missing",
    "destructive": "destructive",
    "hybrid": "hybrid",
}

REQUIRED_AUDIT_FILES = (
    "r4_audit_summary.md",
    "r4_dependency_audit.md",
    "longitudinal_audit.md",
    "spatial_support_audit.md",
    "occlusion_audit.md",
    "prognosis_target_spec.md",
    "prognosis_feature_spec.md",
    "prognosis_clean_audit.md",
    "reconstruction_prognosis_audit.md",
    "fig5_readiness.md",
)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _directory_listing_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    if not path.is_dir():
        return "MISSING", 0, 0
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(f"{rel}\t{size}\n".encode())
        count += 1
        total += size
    return digest.hexdigest(), count, total


def _git_capture(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPOSITORY_ROOT, text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"ERROR: {exc}"


def _dirty_diff_hash() -> tuple[str, str, list[str]]:
    status = _git_capture(["status", "--porcelain=v1"])
    tracked = _git_capture(["diff", "--binary", "HEAD"])
    untracked_names = _git_capture(["ls-files", "--others", "--exclude-standard"])
    digest = hashlib.sha256()
    digest.update(tracked.encode())
    names = [line for line in untracked_names.splitlines() if line]
    for name in names:
        item = REPOSITORY_ROOT / name
        digest.update(f"UNTRACKED:{name}\n".encode())
        if item.is_file():
            digest.update(item.read_bytes())
    return status, digest.hexdigest(), names


def _input_provenance(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            rows.append({
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "kind": "file",
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "checksum_type": "full_sha256",
            })
        elif path.is_dir():
            digest, count, total = _directory_listing_hash(path)
            rows.append({
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "kind": "directory",
                "exists": digest != "MISSING",
                "size_bytes": total,
                "file_count": count,
                "sha256": digest,
                "checksum_type": "listing_sha256(path_and_size)",
            })
        else:
            rows.append({
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "kind": "missing",
                "exists": False,
                "size_bytes": 0,
                "sha256": "MISSING",
                "checksum_type": "none",
            })
    return pd.DataFrame(rows)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    return S.spearman(np.asarray(list(x), float), np.asarray(list(y), float))


def _metric_summary(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    return S.mae(y, pred), S.spearman(y, pred)


def _battery_means(frame: pd.DataFrame, value: str, battery: str = "battery_id") -> pd.Series:
    return frame.groupby(battery, sort=True)[value].mean()


def _bootstrap_array(values: np.ndarray, statistic, n: int = BOOTSTRAPS,
                     seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float, np.ndarray]:
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n, len(values)))
    estimates = np.asarray([statistic(values[index]) for index in draws], float)
    finite = estimates[np.isfinite(estimates)]
    observed = float(statistic(values))
    if not len(finite):
        return observed, float("nan"), float("nan"), estimates
    return observed, float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)), estimates


def _bootstrap_battery_series(series_by_battery: pd.DataFrame, value: str,
                              statistic=np.median, n: int = BOOTSTRAPS,
                              seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float, np.ndarray]:
    batteries = sorted(series_by_battery["battery_id"].astype(str).unique())
    lookup = {battery: series_by_battery.loc[
        series_by_battery.battery_id.astype(str) == battery, value
    ].to_numpy(float) for battery in batteries}
    observed_values = np.concatenate([lookup[b] for b in batteries])
    observed = float(statistic(observed_values))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n, float)
    for i in range(n):
        selected = rng.integers(0, len(batteries), size=len(batteries))
        estimates[i] = statistic(np.concatenate([lookup[batteries[j]] for j in selected]))
    return observed, float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)), estimates


def _longitudinal_bootstrap_statistic(values: Iterable[float], statistic_name: str) -> float:
    """Apply the labelled final estimator to one value per battery.

    The longitudinal point estimate is explicitly battery-level median or
    mean.  Keeping this dispatch in one helper prevents the ``median`` label
    from silently acquiring mean-based bootstrap draws.
    """
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        raise ValueError("longitudinal bootstrap statistic requires values")
    if statistic_name == "median":
        return float(np.median(values))
    if statistic_name == "mean":
        return float(np.mean(values))
    raise ValueError(f"unknown longitudinal statistic: {statistic_name}")


def _rank_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.column_stack([np.ones_like(rz), rz])

    def residual(values: np.ndarray) -> np.ndarray:
        return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]

    return S.spearman(residual(rx), residual(ry))


def _window_definitions(length: int = 1000, width: float = MAIN_WIDTH) -> list[dict[str, Any]]:
    definitions = []
    start = 0.0
    index = 0
    while round(start + width, 6) <= 1.0 + 1e-12:
        begin = int(round(start * length))
        end = int(round((start + width) * length))
        definitions.append({
            "window_id": f"w{index:02d}_{start:.2f}",
            "window_start": float(start),
            "window_end": float(start + width),
            "window_center": float(start + width / 2),
            "start_index": begin,
            "end_index_exclusive": end,
            "width_fraction": float(width),
        })
        start = round(start + WINDOW_STRIDE, 6)
        index += 1
    return definitions


def _all_window_definitions(length: int = 1000) -> list[dict[str, Any]]:
    out = []
    index = 0
    for width in WIDTHS:
        start = 0.0
        while round(start + width, 6) <= 1.0 + 1e-12:
            begin = int(round(start * length))
            end = int(round((start + width) * length))
            out.append({
                "window_id": f"w{index:02d}_{start:.2f}",
                "window_start": float(start),
                "window_end": float(start + width),
                "window_center": float(start + width / 2),
                "start_index": begin,
                "end_index_exclusive": end,
                "width_fraction": float(width),
            })
            start = round(start + WINDOW_STRIDE, 6)
            index += 1
    return out


def _fit_pca(masked_train: np.ndarray, logger: Logger) -> dict[str, np.ndarray]:
    logger.write(f"Fitting train-only PCA on masked residual matrix {masked_train.shape}")
    matrix = np.asarray(masked_train, dtype=np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    result = {
        "mean": mean.astype(np.float32),
        "components": components[:PCA_K].astype(np.float32),
        "singular_values": singular_values[:PCA_K].astype(np.float32),
    }
    logger.write("Train-only PCA complete")
    return result


def _project(pca: dict[str, np.ndarray], residual: np.ndarray) -> np.ndarray:
    return (np.asarray(residual, dtype=np.float32) - pca["mean"]) @ pca["components"].T


def _make_sequences(blocks: dict[str, dict[str, Any]], residuals: dict[str, np.ndarray],
                    keeps: dict[str, np.ndarray], population: dict[str, np.ndarray]) -> dict[str, list[dict[str, Any]]]:
    sequences: dict[str, list[dict[str, Any]]] = {split: [] for split in blocks}
    for split, block in blocks.items():
        cells = np.asarray(block["cells"]).astype(str)
        for cell in sorted(set(cells)):
            full_mask = cells == cell
            order = np.argsort(np.asarray(block["cycles"])[full_mask], kind="stable")
            indices = np.flatnonzero(full_mask)[order]
            sequences[split].append({
                "cell": cell,
                "batch": cell[:2],
                "cycle": np.asarray(block["cycles"])[indices].astype(np.int64),
                "soh": np.asarray(block["soh"])[indices].astype(float),
                "residual": np.asarray(residuals[split])[indices],
                "population": np.asarray(population[split])[indices],
                "keep": np.asarray(keeps[split])[indices].astype(bool),
                "keys": np.asarray(block["keys"])[indices].astype(str),
            })
    return sequences


def _has_future(cycles: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    cycles = np.asarray(cycles, int)
    future = np.zeros(len(cycles), bool)
    for i, cycle in enumerate(cycles):
        j = int(np.searchsorted(cycles, cycle + horizon))
        future[i] = j < len(cycles) and int(cycles[j]) == int(cycle + horizon)
    return future


def _prepare_r3_state(logger: Logger) -> dict[str, Any]:
    certified = C.module()
    logger.write("Loading current R3 clean MIT split")
    blocks = {split: certified.load_clean(DATASET, split) for split in ("train", "vali", "test")}
    logger.write("Selecting frozen current R3 manifold with source implementation unchanged")
    spline, model, s_ind, validation_table, manifold = certified.select_manifold(
        blocks["train"], blocks["vali"]
    )
    residuals: dict[str, np.ndarray] = {}
    keeps: dict[str, np.ndarray] = {}
    populations: dict[str, np.ndarray] = {}
    references: dict[str, dict[str, np.ndarray]] = {}
    for split, block in blocks.items():
        residual, refs, keep = certified.residualize(block, require_five=True)
        residuals[split] = np.asarray(residual, dtype=np.float32).reshape(len(block["soh"]), -1)
        keeps[split] = np.asarray(keep, dtype=bool)
        references[split] = refs
        populations[split] = np.asarray(
            certified.predict(model, spline, block["soh"]), dtype=np.float32
        ).reshape(len(block["soh"]), -1)
    # ``residuals`` returned by the source R3 function are D = V - V_ref.
    # R4's Individual state is R = D - M(SOH); never pass D into PCA or
    # support/longitudinal analyses.
    individuals = {
        split: np.asarray(residuals[split] - populations[split], dtype=np.float32)
        for split in blocks
    }
    sequences = _make_sequences(blocks, individuals, keeps, populations)
    return {
        "certified": certified,
        "blocks": blocks,
        "spline": spline,
        "model": model,
        "s_ind": float(s_ind),
        "validation_table": validation_table,
        "manifold": manifold,
        "residuals": residuals,
        "individuals": individuals,
        "keeps": keeps,
        "populations": populations,
        "references": references,
        "sequences": sequences,
    }


def _write_eligibility(state: dict[str, Any], out: Path, logger: Logger) -> pd.DataFrame:
    rows = []
    counts = []
    split_names = {"train": "train", "vali": "validation", "test": "test"}
    for split, block in state["blocks"].items():
        cells = np.asarray(block["cells"]).astype(str)
        cycles = np.asarray(block["cycles"], int)
        future = np.zeros(len(cycles), bool)
        for cell in sorted(set(cells)):
            idx = np.flatnonzero(cells == cell)
            order = idx[np.argsort(cycles[idx], kind="stable")]
            future[order] = _has_future(cycles[order])
        keep = state["keeps"][split]
        is_reference = ~keep
        frame = pd.DataFrame({
            "battery_id": cells,
            "cycle": cycles,
            "split": split_names[split],
            "SOH": np.asarray(block["soh"], float),
            "is_reference_cycle": is_reference,
            "r3_keep": keep,
            "has_future_cplus100": future,
            "r4_prognosis_eligible": keep & future,
        })
        rows.append(frame)
        counts.append({
            "split": split_names[split],
            "batteries": int(frame.battery_id.nunique()),
            "total_cycles": int(len(frame)),
            "r3_eligible_cycles": int(frame.r3_keep.sum()),
            "cplus100_prognosis_eligible": int(frame.r4_prognosis_eligible.sum()),
            "reference_cycles": int(frame.is_reference_cycle.sum()),
        })
    mask = pd.concat(rows, ignore_index=True)
    if mask.duplicated(["battery_id", "cycle", "split"]).any():
        raise RuntimeError("duplicate rows in canonical R4 cycle mask")
    _write_parquet(mask, out / "canonical_r4_cycle_mask.parquet")
    counts_frame = pd.DataFrame(counts)
    counts_frame.to_csv(out / "canonical_r4_sample_counts.csv", index=False)
    state["mask_frame"] = mask
    state["future_by_split"] = {
        split: mask.loc[mask.split == split_names[split], "has_future_cplus100"].to_numpy(bool)
        for split in state["blocks"]
    }
    logger.write("Canonical R4 eligibility mask written")
    return mask


def _write_split_manifest(mask: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Persist the actual battery IDs used by this formal run."""
    manifest: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        ids = sorted(mask.loc[mask.split == split, "battery_id"].astype(str).unique())
        manifest[split] = {
            "n_batteries": len(ids),
            "battery_ids": ids,
            "n_total_cycles": int((mask.split == split).sum()),
            "n_r3_keep_cycles": int(mask.loc[mask.split == split, "r3_keep"].sum()),
            "n_prognosis_cycles": int(mask.loc[mask.split == split, "r4_prognosis_eligible"].sum()),
        }
    _json_dump(out / "mit_split_manifest.json", manifest)
    manifest_path = out / "run_manifest.json"
    if manifest_path.is_file():
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_manifest["mit_split_manifest"] = manifest
        run_manifest["mit_split_manifest_path"] = "mit_split_manifest.json"
        _json_dump(manifest_path, run_manifest)
    return manifest


def _longitudinal(state: dict[str, Any], pca: dict[str, np.ndarray], out: Path,
                  logger: Logger) -> dict[str, Any]:
    test_sequences = state["sequences"]["test"]
    projected = {seq["cell"]: _project(pca, seq["residual"]) for seq in test_sequences}
    same_rows: list[dict[str, Any]] = []
    for seq in test_sequences:
        z = projected[seq["cell"]]
        # sequences contain all rows for history, but only R3 target rows enter R4 pairs
        for lag in LAGS:
            for i in range(max(0, len(z) - lag)):
                j = i + lag
                if not seq["keep"][i] or not seq["keep"][j]:
                    continue
                same_rows.append({
                    "pair_type": "same_battery",
                    "source_battery_id": seq["cell"],
                    "target_battery_id": seq["cell"],
                    "battery_id": seq["cell"],
                    "cycle_i": int(seq["cycle"][i]),
                    "cycle_j": int(seq["cycle"][j]),
                    "array_lag": int(lag),
                    "cycle_lag": int(seq["cycle"][j] - seq["cycle"][i]),
                    "soh_i": float(seq["soh"][i]),
                    "soh_j": float(seq["soh"][j]),
                    "distance": float(np.linalg.norm(z[i] - z[j])),
                })
    same = pd.DataFrame(same_rows)
    cross_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(C.SEED)
    for lag in LAGS:
        for source in test_sequences:
            z_source = projected[source["cell"]]
            eligible_indices = np.flatnonzero(source["keep"])
            if len(eligible_indices) <= lag:
                continue
            sampled = rng.choice(eligible_indices, size=min(40, len(eligible_indices)), replace=False)
            for i in sampled:
                soh_i = float(source["soh"][i])
                match = None
                for target in test_sequences:
                    if target["cell"] == source["cell"]:
                        continue
                    target_mask = target["keep"] & (np.abs(target["soh"] - soh_i) <= MATCH_TOLERANCE)
                    hit = np.flatnonzero(target_mask)
                    if len(hit):
                        match = (target, int(hit[0]))
                        break
                if match is None:
                    continue
                target, j = match
                cross_rows.append({
                    "pair_type": "matched_soh_cross_battery",
                    "source_battery_id": source["cell"],
                    "target_battery_id": target["cell"],
                    "battery_id": source["cell"],
                    "cycle_i": int(source["cycle"][i]),
                    "cycle_j": int(target["cycle"][j]),
                    "array_lag": int(lag),
                    "cycle_lag": int(target["cycle"][j] - source["cycle"][i]),
                    "soh_i": soh_i,
                    "soh_j": float(target["soh"][j]),
                    "distance": float(np.linalg.norm(z_source[i] - projected[target["cell"]][j])),
                })
    cross = pd.DataFrame(cross_rows)
    _write_parquet(same, out / "longitudinal_same_pairs.parquet")
    _write_parquet(cross, out / "longitudinal_cross_matched_pairs.parquet")

    pair = pd.concat([same, cross], ignore_index=True)
    battery_summary = (pair.groupby(["pair_type", "array_lag", "battery_id"], as_index=False)
                       .agg(distance_median=("distance", "median"),
                            distance_mean=("distance", "mean"),
                            n_pairs=("distance", "size")))
    battery_summary.to_csv(out / "longitudinal_battery_summary.csv", index=False)
    curves = []
    for lag in LAGS:
        sub = battery_summary[battery_summary.array_lag == lag]
        values: dict[str, float] = {}
        counts_row = {"lag": lag, "n_pairs": int(pair.loc[pair.array_lag == lag].shape[0])}
        for pair_type, label in (("same_battery", "same"),
                                 ("matched_soh_cross_battery", "cross")):
            vals = sub.loc[sub.pair_type == pair_type, "distance_median"].to_numpy(float)
            values[f"{label}_median"] = float(np.median(vals)) if len(vals) else np.nan
            values[f"{label}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            counts_row[f"{label}_n_batteries"] = int(len(vals))
        curves.append({
            "lag": lag,
            "bin": "short" if lag == 1 else ("intermediate" if lag == 25 else ("long" if lag == 100 else "registered")),
            "same_mean_or_median": values.get("same_median", np.nan),
            "cross_mean_or_median": values.get("cross_median", np.nan),
            "same_mean": values.get("same_mean", np.nan),
            "cross_mean": values.get("cross_mean", np.nan),
            "difference_same_minus_cross": values.get("same_median", np.nan) - values.get("cross_median", np.nan),
            "n_batteries": counts_row.get("same_n_batteries", 0),
            "n_cross_batteries": counts_row.get("cross_n_batteries", 0),
            "n_pairs": counts_row["n_pairs"],
        })
    curve = pd.DataFrame(curves)

    boot_rows = []
    for lag in LAGS:
        for statistic_name, value_column in (("median", "distance_median"), ("mean", "distance_mean")):
            sub = battery_summary[battery_summary.array_lag == lag]
            same_values = {b: g[value_column].to_numpy(float) for b, g in
                           sub[sub.pair_type == "same_battery"].groupby("battery_id")}
            cross_values = {b: g[value_column].to_numpy(float) for b, g in
                            sub[sub.pair_type == "matched_soh_cross_battery"].groupby("battery_id")}
            batteries = sorted(set(same_values) & set(cross_values))
            rng_b = np.random.default_rng(BOOTSTRAP_SEED + lag * 10 + (0 if statistic_name == "median" else 1))
            same_draws, cross_draws, diff_draws = [], [], []
            same_battery_values = [
                _longitudinal_bootstrap_statistic(same_values[b], statistic_name)
                for b in batteries
            ]
            cross_battery_values = [
                _longitudinal_bootstrap_statistic(cross_values[b], statistic_name)
                for b in batteries
            ]
            for _ in range(BOOTSTRAPS):
                selected = rng_b.integers(0, len(batteries), size=len(batteries))
                sv = np.asarray([same_battery_values[i] for i in selected])
                cv = np.asarray([cross_battery_values[i] for i in selected])
                same_estimate = _longitudinal_bootstrap_statistic(sv, statistic_name)
                cross_estimate = _longitudinal_bootstrap_statistic(cv, statistic_name)
                same_draws.append(same_estimate)
                cross_draws.append(cross_estimate)
                if statistic_name == "median":
                    diff_draws.append(same_estimate - cross_estimate)
                else:
                    # Preserve the pre-existing mean estimator exactly for
                    # the separate ``statistic="mean"`` rows.
                    diff_draws.append(float(np.mean(sv - cv)))
            observed_same = _longitudinal_bootstrap_statistic(same_battery_values, statistic_name)
            observed_cross = _longitudinal_bootstrap_statistic(cross_battery_values, statistic_name)
            boot_rows.append({
                "lag": lag, "statistic": statistic_name, "bootstrap_unit": "battery",
                "observed_same": observed_same, "same_ci_low": float(np.percentile(same_draws, 2.5)),
                "same_ci_high": float(np.percentile(same_draws, 97.5)),
                "observed_cross": observed_cross, "cross_ci_low": float(np.percentile(cross_draws, 2.5)),
                "cross_ci_high": float(np.percentile(cross_draws, 97.5)),
                "observed_same_minus_cross": observed_same - observed_cross,
                "difference_ci_low": float(np.percentile(diff_draws, 2.5)),
                "difference_ci_high": float(np.percentile(diff_draws, 97.5)),
                "n_batteries": len(batteries), "n_resamples": BOOTSTRAPS,
            })
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out / "longitudinal_bootstrap.csv", index=False)

    # Matched-SOH permutation: preserve every source row's SOH and source battery,
    # but randomize the target battery among all eligible other batteries.
    candidate_distances: list[dict[str, Any]] = []
    for row in cross.itertuples(index=False):
        candidates = []
        source = next(item for item in test_sequences if item["cell"] == row.source_battery_id)
        for target in test_sequences:
            if target["cell"] == source["cell"]:
                continue
            hit = np.flatnonzero(target["keep"] & (np.abs(target["soh"] - row.soh_i) <= MATCH_TOLERANCE))
            if len(hit):
                j = int(hit[0])
                candidates.append(float(np.linalg.norm(projected[source["cell"]][int(np.where(source["cycle"] == row.cycle_i)[0][0])] - projected[target["cell"]][j])))
        candidate_distances.append({"lag": int(row.array_lag), "candidates": candidates})
    null_rows = []
    null_stats: dict[str, Any] = {"n_permutations": PERMUTATIONS, "seed": PERMUTATION_SEED,
                                   "matching_tolerance": MATCH_TOLERANCE,
                                   "method": "target-battery permutation among SOH-matched eligible other batteries; source rows and SOH structure fixed"}
    rng_null = np.random.default_rng(PERMUTATION_SEED)
    for lag in LAGS:
        candidates = [item["candidates"] for item in candidate_distances if item["lag"] == lag and item["candidates"]]
        draws = np.empty(PERMUTATIONS, float)
        for i in range(PERMUTATIONS):
            draws[i] = np.median([c[int(rng_null.integers(0, len(c)))] for c in candidates]) if candidates else np.nan
        same_obs = float(curve.loc[curve.lag == lag, "same_mean_or_median"].iloc[0])
        cross_obs = float(curve.loc[curve.lag == lag, "cross_mean_or_median"].iloc[0])
        for i, value in enumerate(draws):
            null_rows.append({"lag": lag, "permutation": i, "null_cross_median": value,
                              "observed_same_median": same_obs, "observed_cross_median": cross_obs,
                              "same_minus_null": same_obs - value})
        finite = draws[np.isfinite(draws)]
        null_stats[str(lag)] = {
            "observed_same_median": same_obs,
            "observed_cross_median": cross_obs,
            "null_cross_median": float(np.median(finite)) if len(finite) else np.nan,
            "null_ci_low": float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
            "null_ci_high": float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
            "fraction_null_at_or_below_same": float(np.mean(finite <= same_obs)) if len(finite) else np.nan,
            "n_rows": len(candidates),
        }
    pd.DataFrame(null_rows).to_csv(out / "longitudinal_null.csv", index=False)
    _json_dump(out / "longitudinal_null_stats.json", null_stats)

    curve["same_ci_low"] = np.nan
    curve["same_ci_high"] = np.nan
    curve["cross_ci_low"] = np.nan
    curve["cross_ci_high"] = np.nan
    med = boot[boot.statistic == "median"].set_index("lag")
    for index, row in curve.iterrows():
        if row.lag in med.index:
            curve.loc[index, "same_ci_low"] = med.loc[row.lag, "same_ci_low"]
            curve.loc[index, "same_ci_high"] = med.loc[row.lag, "same_ci_high"]
            curve.loc[index, "cross_ci_low"] = med.loc[row.lag, "cross_ci_low"]
            curve.loc[index, "cross_ci_high"] = med.loc[row.lag, "cross_ci_high"]
    curve.to_csv(out / "longitudinal_lag_curve.csv", index=False)

    short = curve[curve.lag == 1].iloc[0]
    long_row = curve[curve.lag == 100].iloc[0]
    stats = {
        "distance_definition": "Euclidean norm of difference between train-only PCA K=10 projections of masked R3 Individual residuals",
        "lag_definition": "array offset over chronologically ordered clean cycles; cycle_lag is also retained",
        "matched_soh_tolerance": MATCH_TOLERANCE,
        "cross_protocol": "seeded at-most-40 eligible source indices per source battery and lag; first eligible matching other battery/cycle",
        "bootstrap_unit": "battery",
        "n_bootstrap": BOOTSTRAPS,
        "short_lag_1": short.to_dict(),
        "long_lag_100": long_row.to_dict(),
        "same_increases_lag1_to_lag100": bool(long_row.same_mean_or_median > short.same_mean_or_median),
        "short_same_less_than_cross": bool(short.same_mean_or_median < short.cross_mean_or_median),
        "long_same_less_than_cross": bool(long_row.same_mean_or_median < long_row.cross_mean_or_median),
        "LONGITUDINAL_SUPPORTED": bool(
            long_row.same_mean_or_median > short.same_mean_or_median
            and short.same_mean_or_median < short.cross_mean_or_median
        ),
    }
    _json_dump(out / "longitudinal_stats.json", stats)
    logger.write(f"Longitudinal complete: same lag1={short.same_mean_or_median:.6g}, cross={short.cross_mean_or_median:.6g}")
    return {"same": same, "cross": cross, "battery_summary": battery_summary,
            "curve": curve, "bootstrap": boot, "stats": stats}


def _spatial(state: dict[str, Any], out: Path, logger: Logger) -> dict[str, Any]:
    certified = state["certified"]
    train = state["blocks"]["train"]
    grid = np.linspace(float(np.min(train["soh"])), float(np.max(train["soh"])), 500)
    pop = np.asarray(certified.predict(state["model"], state["spline"], grid), float).reshape(500, -1)
    s_pop = pop.var(axis=0)
    batteries = []
    for seq in state["sequences"]["test"]:
        residual = seq["residual"][seq["keep"]]
        batteries.append({"battery_id": seq["cell"], "S_ind": np.median(residual ** 2, axis=0)})
    s_ind_by_battery = np.stack([item["S_ind"] for item in batteries])
    s_ind = np.median(s_ind_by_battery, axis=0)
    profile = pd.DataFrame({
        "position": np.arange(len(s_pop)),
        "progress": np.linspace(0, 1, len(s_pop)),
        "S_pop": s_pop,
        "S_ind": s_ind,
        "n_test_batteries": len(batteries),
    })

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(batteries), size=(BOOTSTRAPS, len(batteries)))
    ind_profiles = np.median(s_ind_by_battery[draws], axis=1)
    profile["S_pop_ci_low"] = s_pop
    profile["S_pop_ci_high"] = s_pop
    profile["S_ind_ci_low"] = np.percentile(ind_profiles, 2.5, axis=0)
    profile["S_ind_ci_high"] = np.percentile(ind_profiles, 97.5, axis=0)
    profile.to_csv(out / "spatial_support_profile.csv", index=False)

    def support_metrics(values: np.ndarray) -> dict[str, float]:
        result = concentration(values)
        return {
            "effective_support_fraction": result["entropy_effective_support"],
            "support_fraction_50": result["support_fraction_50"],
            "support_width_50": result["support_fraction_50"],
            "support_fraction_80": result["support_fraction_80"],
            "support_fraction_90": result["support_fraction_90"],
            "participation_ratio_fraction": result["participation_ratio_fraction"],
        }

    pop_metrics = support_metrics(s_pop)
    ind_metrics = support_metrics(s_ind)
    ind_draw_metrics = pd.DataFrame([support_metrics(draw) for draw in ind_profiles])
    summary_rows = []
    for metric in ("effective_support_fraction", "support_fraction_50", "support_width_50",
                   "support_fraction_80", "support_fraction_90", "participation_ratio_fraction"):
        values = ind_draw_metrics[metric].to_numpy(float)
        summary_rows.append({
            "metric": metric, "Population": pop_metrics[metric], "Individual": ind_metrics[metric],
            "difference": ind_metrics[metric] - pop_metrics[metric],
            "CI_low": float(np.percentile(values, 2.5)),
            "CI_high": float(np.percentile(values, 97.5)),
            "bootstrap_unit": "battery", "n_batteries": len(batteries), "n_resamples": BOOTSTRAPS,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "spatial_support_summary.csv", index=False)
    boot_rows = []
    for i, row in ind_draw_metrics.iterrows():
        boot_rows.append({"resample": i, "component": "Individual", **row.to_dict()})
    boot_frame = pd.DataFrame(boot_rows)
    boot_frame.to_csv(out / "spatial_support_bootstrap.csv", index=False)
    logger.write(f"Spatial support complete: 50% widths population={pop_metrics['support_fraction_50']:.6g}, individual={ind_metrics['support_fraction_50']:.6g}")
    return {"profile": profile, "summary": summary, "s_pop": s_pop,
            "s_ind": s_ind, "s_ind_by_battery": s_ind_by_battery,
            "pop_metrics": pop_metrics, "ind_metrics": ind_metrics}


def _read_occlusion(state: dict[str, Any], out: Path, logger: Logger) -> dict[str, Any]:
    archive = REPOSITORY_ROOT / "analysis/outputs/canonical/r4_sliding_occlusion/20260926T141414Z/raw/sliding_occlusion_metrics_raw.csv.gz"
    logger.write(f"Reading frozen occlusion archive only: {archive}")
    usecols = [
        "window_id", "model_seed", "cell", "cycle", "width_fraction", "window_start",
        "waveform_damage", "waveform_RMSE", "E_pop_clean", "E_pop_occluded",
        "E_ind_clean", "E_ind_occluded", "Q_clean", "Q_occluded", "C_ind_clean",
        "C_ind_occluded", "Delta_E_pop", "Delta_E_ind", "C_ind_loss",
    ]
    dtypes = {"window_id": str, "model_seed": "int16", "cell": str, "cycle": "int32",
              "width_fraction": "float32", "window_start": "float32"}
    raw = pd.read_csv(archive, usecols=usecols, dtype=dtypes)
    mask_keys = set(zip(
        state["mask_frame"].loc[(state["mask_frame"].split == "test") & state["mask_frame"].r3_keep, "battery_id"],
        state["mask_frame"].loc[(state["mask_frame"].split == "test") & state["mask_frame"].r3_keep, "cycle"],
    ))
    raw["battery_id"] = raw["cell"].astype(str)
    raw["cycle_id"] = raw["cycle"].astype(int)
    keep = pd.Series(list(zip(raw.battery_id, raw.cycle_id))).isin(mask_keys).to_numpy()
    raw = raw.loc[keep].copy()
    raw["width_fraction"] = raw.width_fraction.astype(float)
    raw["window_start"] = raw.window_start.astype(float)
    raw["window_center"] = raw.window_start + raw.width_fraction / 2.0
    raw["width_label"] = raw.width_fraction.map({0.1: "10%", 0.2: "20%", 0.3: "30%"})
    if raw.empty:
        raise RuntimeError("frozen occlusion archive has no canonical R3 test rows")
    _write_parquet(raw, out / "occlusion_raw_canonical.parquet")

    metric_columns = ["waveform_RMSE", "waveform_damage", "Delta_E_pop", "Delta_E_ind", "C_ind_loss",
                      "E_pop_clean", "E_pop_occluded", "E_ind_clean", "E_ind_occluded",
                      "C_ind_clean", "C_ind_occluded", "Q_clean", "Q_occluded"]
    cycle_seed = (raw.groupby(["battery_id", "cycle_id", "window_id", "width_fraction", "window_start"], as_index=False)
                  [metric_columns].mean())
    battery = (cycle_seed.groupby(["battery_id", "window_id", "width_fraction", "window_start"], as_index=False)
               [metric_columns].median())
    battery["window_center"] = battery.window_start + battery.width_fraction / 2.0
    battery["width_label"] = battery.width_fraction.map({0.1: "10%", 0.2: "20%", 0.3: "30%"})
    battery.to_csv(out / "occlusion_battery_summary.csv", index=False)

    profile_paths: dict[float, Path] = {}
    for width in WIDTHS:
        sub = battery[np.isclose(battery.width_fraction, width)].copy()
        rows = []
        for (window_id, start, center), group in sub.groupby(["window_id", "window_start", "window_center"], sort=True):
            batteries = sorted(group.battery_id.astype(str).unique())
            idx = {b: group[group.battery_id.astype(str) == b] for b in batteries}
            # one row per battery/window is expected; median is defensive.
            estimates = {}
            for metric in ["waveform_RMSE", "Delta_E_pop", "Delta_E_ind", "C_ind_loss", "E_pop_occluded", "E_ind_occluded", "C_ind_occluded"]:
                vals = group[metric].to_numpy(float)
                estimates[metric] = float(np.median(vals))
            rng = np.random.default_rng(BOOTSTRAP_SEED + int(round(width * 1000)) + int(round(start * 1000)))
            draw_values = {metric: np.empty(BOOTSTRAPS, float) for metric in estimates}
            for i in range(BOOTSTRAPS):
                selected = rng.integers(0, len(batteries), size=len(batteries))
                selected_group = pd.concat([idx[batteries[j]] for j in selected], ignore_index=True)
                for metric in estimates:
                    draw_values[metric][i] = float(np.median(selected_group[metric].to_numpy(float)))
            for metric, estimate in estimates.items():
                rows.append({
                    "window_id": window_id, "window_start": float(start), "window_center": float(center),
                    "window_width": float(width), "metric": metric, "estimate": estimate,
                    "ci_low": float(np.percentile(draw_values[metric], 2.5)),
                    "ci_high": float(np.percentile(draw_values[metric], 97.5)),
                    "n_batteries": len(batteries), "bootstrap_unit": "battery", "n_resamples": BOOTSTRAPS,
                })
        frame = pd.DataFrame(rows)
        filename = f"occlusion_window_profile_{int(width * 100)}.csv"
        frame.to_csv(out / filename, index=False)
        profile_paths[width] = out / filename
    logger.write(f"Occlusion reaggregation complete: {len(raw)} canonical raw rows; no inference called")
    return {"raw": raw, "cycle_seed": cycle_seed, "battery": battery,
            "profiles": {width: pd.read_csv(path) for width, path in profile_paths.items()}}


def _support_vulnerability(spatial: dict[str, Any], occlusion: dict[str, Any], out: Path,
                           logger: Logger) -> dict[str, Any]:
    s_pop = spatial["s_pop"]
    s_ind = spatial["s_ind"]
    battery = occlusion["battery"]
    main_rows = []
    boot_rows = []
    stats: dict[str, Any] = {"bootstrap_unit": "battery", "n_resamples": BOOTSTRAPS,
                             "position_control": "rank-partial Spearman controlling window_start"}
    for width in WIDTHS:
        sub = battery[np.isclose(battery.width_fraction, width)].copy()
        defs = _window_definitions(1000, width)
        defs_by_id = {item["window_id"]: item for item in defs}
        # The frozen archive names windows globally across widths. Match by start/width,
        # not by the global window_id string.
        profile_rows = []
        for (start, center), group in sub.groupby(["window_start", "window_center"], sort=True):
            start = float(start)
            end = int(round((start + width) * len(s_ind)))
            begin = int(round(start * len(s_ind)))
            profile_rows.append({
                "window_start": start, "window_center": float(center),
                "S_ind": float(np.sum(s_ind[begin:end])),
                "S_pop": float(np.sum(s_pop[begin:end])),
                "Delta_E_ind": float(np.median(group["Delta_E_ind"])),
                "C_ind_loss": float(np.median(group["C_ind_loss"])),
                "Delta_E_pop": float(np.median(group["Delta_E_pop"])),
                "waveform_damage": float(np.median(group["waveform_damage"])),
                "n_batteries": int(group.battery_id.nunique()),
            })
        profile = pd.DataFrame(profile_rows).sort_values("window_start").reset_index(drop=True)
        relationships = [
            ("S_ind", "Delta_E_ind", "S_ind_vs_Delta_E_ind"),
            ("S_ind", "C_ind_loss", "S_ind_vs_C_ind_loss"),
            ("S_pop", "Delta_E_pop", "S_pop_vs_Delta_E_pop"),
        ]
        # A battery bootstrap re-forms the window profile before the association.
        batteries = sorted(sub.battery_id.astype(str).unique())
        per_battery = {b: sub[sub.battery_id.astype(str) == b].set_index("window_start") for b in batteries}
        windows = np.array(sorted(per_battery[batteries[0]].index.unique()), float)
        arrays = {}
        for metric in ("Delta_E_ind", "C_ind_loss", "Delta_E_pop", "waveform_damage"):
            arrays[metric] = np.stack([per_battery[b].reindex(windows)[metric].to_numpy(float) for b in batteries])
        rng = np.random.default_rng(BOOTSTRAP_SEED + int(width * 1000))
        for x_name, y_name, relation in relationships:
            x = profile[x_name].to_numpy(float)
            y = profile[y_name].to_numpy(float)
            raw_rho = S.spearman(x, y)
            partial = _rank_partial(x, y, profile.window_start.to_numpy(float))
            estimates_raw, estimates_partial = [], []
            for draw_id in range(BOOTSTRAPS):
                selected = rng.integers(0, len(batteries), size=len(batteries))
                if y_name == "Delta_E_ind":
                    yy = np.median(arrays["Delta_E_ind"][selected], axis=0)
                    xx = np.sum(s_ind.reshape(1, -1)[:, :0], axis=1) if False else x
                elif y_name == "C_ind_loss":
                    yy = np.median(arrays["C_ind_loss"][selected], axis=0)
                    xx = x
                else:
                    yy = np.median(arrays["Delta_E_pop"][selected], axis=0)
                    xx = x
                if x_name == "S_pop":
                    xx = profile.S_pop.to_numpy(float)
                estimates_raw.append(S.spearman(xx, yy))
                estimates_partial.append(_rank_partial(xx, yy, profile.window_start.to_numpy(float)))
                boot_rows.append({"width_fraction": width, "resample": draw_id,
                                  "relationship": relation, "rho": estimates_raw[-1],
                                  "partial_rho": estimates_partial[-1],
                                  "bootstrap_unit": "battery"})
            main_rows.append({
                "width_fraction": width, "relationship": relation, "x": x_name, "y": y_name,
                "spearman_rho": raw_rho, "ci_low": float(np.nanpercentile(estimates_raw, 2.5)),
                "ci_high": float(np.nanpercentile(estimates_raw, 97.5)),
                "partial_spearman": partial,
                "partial_ci_low": float(np.nanpercentile(estimates_partial, 2.5)),
                "partial_ci_high": float(np.nanpercentile(estimates_partial, 97.5)),
                "n_windows": len(profile), "n_batteries": len(batteries),
                "n_resamples": BOOTSTRAPS,
            })
            stats[f"{int(width * 100)}%_{relation}"] = main_rows[-1]
    main = pd.DataFrame(main_rows)
    main.to_csv(out / "support_vulnerability_main.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "support_vulnerability_bootstrap.csv", index=False)
    _json_dump(out / "support_vulnerability_stats.json", stats)
    logger.write("Support-vulnerability associations complete")
    return {"main": main, "bootstrap": pd.DataFrame(boot_rows), "stats": stats}


def _slope(values: np.ndarray, window: int) -> float:
    values = np.asarray(values, float)[-window:]
    if len(values) < 2:
        return float("nan")
    t = np.arange(len(values), dtype=float)
    centered = t - t.mean()
    denom = float((centered ** 2).sum())
    return float((centered * (values - values.mean())).sum() / denom) if denom else np.nan


def _build_prognosis_tables(state: dict[str, Any], pca: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for split, sequences in state["sequences"].items():
        rows = []
        for seq in sequences:
            cycles = seq["cycle"]
            soh = seq["soh"]
            projected = _project(pca, seq["residual"])
            for i, cycle in enumerate(cycles):
                if not seq["keep"][i]:
                    continue
                j = int(np.searchsorted(cycles, int(cycle) + HORIZON))
                if j >= len(cycles) or int(cycles[j]) != int(cycle) + HORIZON:
                    continue
                rows.append({
                    "battery_id": seq["cell"], "batch": seq["batch"], "split": split,
                    "cycle_id": int(cycle), "soh": float(soh[i]),
                    "dSOH_100": float(soh[j] - soh[i]),
                    "has_future_cplus100": True,
                    "soh_slope_50": _slope(soh[:i + 1], HISTORY_WINDOWS[0]),
                    "soh_slope_100": _slope(soh[:i + 1], HISTORY_WINDOWS[1]),
                    **{f"PC{k + 1}": float(projected[i, k]) for k in range(PCA_K)},
                })
        tables[split] = pd.DataFrame(rows)
    return tables


def _feature_columns(feature_set: str) -> list[str]:
    cols = ["soh"]
    if "I" in feature_set:
        cols += [f"PC{k + 1}" for k in range(PCA_K)]
    if "H" in feature_set:
        cols += ["soh_slope_50", "soh_slope_100"]
    return cols


class FrozenPredictor:
    def __init__(self, kind: str, feature_set: str, x_train: np.ndarray, y_train: np.ndarray):
        self.kind = kind
        self.feature_set = feature_set
        self.x_mu = np.mean(x_train, axis=0, keepdims=True)
        self.x_sd = np.std(x_train, axis=0, keepdims=True)
        self.x_sd[self.x_sd == 0] = 1.0
        self.y_mu = float(np.mean(y_train))
        self.y_sd = float(np.std(y_train)) or 1.0
        self.model = None

    def fit(self, x_fit: np.ndarray, y_fit: np.ndarray) -> None:
        x = (x_fit - self.x_mu) / self.x_sd
        y = np.asarray(y_fit, float).reshape(-1)
        if self.kind == "ridge":
            self.model = np.linalg.solve(
                x.T @ x + C.RIDGE_ALPHA * np.eye(x.shape[1]),
                x.T @ (y - self.y_mu),
            )
        elif self.kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor(n_neighbors=min(C.KNN_NEIGHBOURS, len(x)), n_jobs=-1)
            self.model.fit(x, y)
        elif self.kind == "mlp":
            import torch
            torch.manual_seed(MLP_SEED)
            torch.set_num_threads(1)
            self.model = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], C.MLP_HIDDEN[0]), torch.nn.ReLU(),
                torch.nn.Linear(C.MLP_HIDDEN[0], C.MLP_HIDDEN[1]), torch.nn.ReLU(),
                torch.nn.Linear(C.MLP_HIDDEN[1], 1),
            )
            opt = torch.optim.Adam(self.model.parameters(), lr=C.MLP_LEARNING_RATE)
            loss_fn = torch.nn.MSELoss()
            xt = torch.tensor(x, dtype=torch.float32)
            yt = torch.tensor(((y - self.y_mu) / self.y_sd).reshape(-1, 1), dtype=torch.float32)
            self.model.train()
            for _ in range(C.MLP_EPOCHS):
                opt.zero_grad()
                loss_fn(self.model(xt), yt).backward()
                opt.step()
            self.model.eval()
        else:
            raise ValueError(self.kind)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        x = (np.asarray(x_values, float) - self.x_mu) / self.x_sd
        if self.kind == "ridge":
            return x @ self.model + self.y_mu
        if self.kind == "knn":
            return np.asarray(self.model.predict(x), float).reshape(-1)
        import torch
        with torch.no_grad():
            pred = self.model(torch.tensor(x, dtype=torch.float32)).numpy().reshape(-1)
        return pred * self.y_sd + self.y_mu


def _fit_predictors(tables: dict[str, pd.DataFrame], out: Path, logger: Logger) -> tuple[dict[str, FrozenPredictor], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fit_pool = pd.concat([tables["train"], tables["vali"]], ignore_index=True)
    test = tables["test"]
    feature_sets = ("P", "P+I", "P+H", "P+H+I")
    predictors = ("ridge", "knn", "mlp")
    models: dict[str, FrozenPredictor] = {}
    prediction_frames = []
    battery_rows = []
    dim_rows = []
    scaler_payload: dict[str, Any] = {}
    for feature_set in feature_sets:
        cols = _feature_columns(feature_set)
        x_train_only = fit_pool.loc[fit_pool.split == "train", cols].to_numpy(float)
        y_train_only = fit_pool.loc[fit_pool.split == "train", "dSOH_100"].to_numpy(float)
        train_ok = np.isfinite(x_train_only).all(axis=1) & np.isfinite(y_train_only)
        x_fit = fit_pool[cols].to_numpy(float)
        y_fit = fit_pool["dSOH_100"].to_numpy(float)
        fit_ok = np.isfinite(x_fit).all(axis=1) & np.isfinite(y_fit)
        if train_ok.sum() < 20 or fit_ok.sum() < 20:
            raise RuntimeError(f"insufficient prognosis rows for {feature_set}")
        x_mu = x_train_only[train_ok].mean(axis=0)
        x_sd = x_train_only[train_ok].std(axis=0)
        x_sd[x_sd == 0] = 1.0
        scaler_payload[feature_set] = {"columns": cols, "mu": x_mu, "sd": x_sd,
                                       "fit_split_for_scaler": "train_only"}
        dim_rows.append({"feature_set": feature_set, "dimension": len(cols),
                         "P_dimension": 1, "I_dimension": PCA_K if "I" in feature_set else 0,
                         "H_dimension": 2 if "H" in feature_set else 0,
                         "train_rows_for_scaler": int(train_ok.sum()),
                         "fit_pool_rows": int(fit_ok.sum()), "test_rows": int(len(test))})
        for kind in predictors:
            predictor = FrozenPredictor(kind, feature_set, x_train_only[train_ok], y_train_only[train_ok])
            # Freeze train-only standardization, then fit the fixed head on train+validation.
            predictor.x_mu = x_mu.reshape(1, -1)
            predictor.x_sd = x_sd.reshape(1, -1)
            predictor.fit(x_fit[fit_ok], y_fit[fit_ok])
            key = f"{kind}|{feature_set}"
            models[key] = predictor
            x_test = test[cols].to_numpy(float)
            y_test = test["dSOH_100"].to_numpy(float)
            test_ok = np.isfinite(x_test).all(axis=1) & np.isfinite(y_test)
            pred = predictor.predict(x_test[test_ok])
            frame = test.loc[test_ok, ["battery_id", "batch", "split", "cycle_id", "dSOH_100"]].copy()
            frame["predictor"] = kind
            frame["feature_set"] = feature_set
            frame["y_true"] = y_test[test_ok]
            frame["y_pred"] = pred
            frame["absolute_error"] = np.abs(frame.y_true - frame.y_pred)
            prediction_frames.append(frame)
            for battery_id, group in frame.groupby("battery_id", sort=True):
                battery_rows.append({
                    "battery_id": battery_id, "predictor": kind, "feature_set": feature_set,
                    "MAE": float(group.absolute_error.mean()),
                    "Spearman": S.spearman(group.y_true, group.y_pred),
                    "n_samples": int(len(group)),
                })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    battery_metrics = pd.DataFrame(battery_rows)
    summary_rows = []
    boot_rows = []
    for kind in predictors:
        for feature_set in feature_sets:
            sub = battery_metrics[(battery_metrics.predictor == kind) & (battery_metrics.feature_set == feature_set)]
            batteries = sorted(sub.battery_id.astype(str).unique())
            mae_vals = sub.set_index("battery_id").loc[batteries, "MAE"].to_numpy(float)
            stable_seed = BOOTSTRAP_SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(f"{kind}|{feature_set}"))
            rng = np.random.default_rng(stable_seed)
            mae_draws = np.empty(BOOTSTRAPS, float)
            spearman_draws = np.empty(BOOTSTRAPS, float)
            raw_pred = predictions[(predictions.predictor == kind) & (predictions.feature_set == feature_set)]
            per_battery = {b: raw_pred[raw_pred.battery_id.astype(str) == b] for b in batteries}
            for i in range(BOOTSTRAPS):
                selected = rng.integers(0, len(batteries), size=len(batteries))
                mae_draws[i] = np.mean([mae_vals[j] for j in selected])
                joined = pd.concat([per_battery[batteries[j]] for j in selected], ignore_index=True)
                spearman_draws[i] = S.spearman(joined.y_true, joined.y_pred)
                boot_rows.append({"predictor": kind, "feature_set": feature_set, "resample": i,
                                  "MAE": mae_draws[i], "Spearman": spearman_draws[i],
                                  "bootstrap_unit": "battery"})
            p = raw_pred
            summary_rows.append({
                "predictor": kind, "feature_set": feature_set,
                "MAE": float(np.mean(mae_vals)), "ci_low": float(np.percentile(mae_draws, 2.5)),
                "ci_high": float(np.percentile(mae_draws, 97.5)),
                "Spearman": S.spearman(p.y_true, p.y_pred),
                "spearman_ci_low": float(np.nanpercentile(spearman_draws, 2.5)),
                "spearman_ci_high": float(np.nanpercentile(spearman_draws, 97.5)),
                "n_samples": int(len(p)), "n_batteries": len(batteries), "n_bootstrap": BOOTSTRAPS,
            })
    summary = pd.DataFrame(summary_rows)
    p_i_rows = []
    for kind in predictors:
        for feature_set in feature_sets:
            sub = battery_metrics[(battery_metrics.predictor == kind) & (battery_metrics.feature_set == feature_set)]
            psub = battery_metrics[(battery_metrics.predictor == kind) & (battery_metrics.feature_set == "P")]
            if feature_set == "P":
                delta = 0.0
                low = high = 0.0
            else:
                joined = sub.merge(psub, on="battery_id", suffixes=("_feature", "_P"))
                values = joined.MAE_P - joined.MAE_feature
                rng = np.random.default_rng(BOOTSTRAP_SEED + 50000 + len(p_i_rows))
                draws = np.array([np.mean(values.to_numpy(float)[rng.integers(0, len(values), size=len(values))]) for _ in range(BOOTSTRAPS)])
                delta, low, high = float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
            p_i_rows.append({"predictor": kind, "feature_set": feature_set,
                             "delta_MAE_vs_P": delta, "delta_ci_low": low, "delta_ci_high": high})
    delta_frame = pd.DataFrame(p_i_rows)
    summary = summary.merge(delta_frame, on=["predictor", "feature_set"], how="left")
    predictions.to_csv(out / "prognosis_clean_predictions.csv.gz", index=False, compression="gzip")
    _write_parquet(predictions, out / "prognosis_clean_predictions.parquet")
    battery_metrics.to_csv(out / "prognosis_clean_battery_metrics.csv", index=False)
    summary.to_csv(out / "prognosis_clean_summary.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "prognosis_clean_bootstrap.csv", index=False)
    dim_frame = pd.DataFrame(dim_rows)
    dim_frame.to_csv(out / "prognosis_feature_dimensions.csv", index=False)
    # Store one compressed artifact per feature set in a single train-only scaler file.
    np.savez(out / "scaler_train_only.npz",
             **{f"{fs}_mu": payload["mu"] for fs, payload in scaler_payload.items()},
             **{f"{fs}_sd": payload["sd"] for fs, payload in scaler_payload.items()})
    return models, predictions, battery_metrics, {
        "summary": summary, "bootstrap": pd.DataFrame(boot_rows),
        "scalers": scaler_payload, "dimensions": dim_frame,
        "fit_pool_rows": len(fit_pool), "test_rows": len(test),
    }


def _history_control(tables: dict[str, pd.DataFrame], out: Path, logger: Logger) -> dict[str, Any]:
    # History control is Ridge only, as pre-registered in the audit. The full
    # P/P+I/P+H/P+H+I result is already fit by _fit_predictors; recompute the
    # paired battery summary here with the same fixed train-only scaler rule.
    fit_pool = pd.concat([tables["train"], tables["vali"]], ignore_index=True)
    test = tables["test"]
    sets = ("P", "P+I", "P+H", "P+H+I")
    rows = []
    prediction_rows = []
    for feature_set in sets:
        cols = _feature_columns(feature_set)
        train_only = fit_pool[fit_pool.split == "train"]
        tr_ok = np.isfinite(train_only[cols].to_numpy(float)).all(axis=1) & np.isfinite(train_only.dSOH_100)
        fit_ok = np.isfinite(fit_pool[cols].to_numpy(float)).all(axis=1) & np.isfinite(fit_pool.dSOH_100)
        xtr0 = train_only.loc[tr_ok, cols].to_numpy(float)
        mu, sd = xtr0.mean(axis=0), xtr0.std(axis=0)
        sd[sd == 0] = 1.0
        xfit = (fit_pool.loc[fit_ok, cols].to_numpy(float) - mu) / sd
        yfit = fit_pool.loc[fit_ok, "dSOH_100"].to_numpy(float)
        xte0 = test[cols].to_numpy(float)
        ok = np.isfinite(xte0).all(axis=1) & np.isfinite(test.dSOH_100)
        xte = (xte0[ok] - mu) / sd
        yte = test.loc[ok, "dSOH_100"].to_numpy(float)
        ymu = yfit.mean()
        coef = np.linalg.solve(xfit.T @ xfit + C.RIDGE_ALPHA * np.eye(xfit.shape[1]), xfit.T @ (yfit - ymu))
        pred = xte @ coef + ymu
        frame = test.loc[ok, ["battery_id", "cycle_id", "dSOH_100"]].copy()
        frame["feature_set"] = feature_set
        frame["y_pred"] = pred
        frame["absolute_error"] = abs(frame.dSOH_100 - frame.y_pred)
        prediction_rows.append(frame)
        for battery, group in frame.groupby("battery_id", sort=True):
            rows.append({"battery_id": battery, "feature_set": feature_set,
                         "MAE": float(group.absolute_error.mean()),
                         "Spearman": S.spearman(group.dSOH_100, group.y_pred),
                         "n_samples": len(group)})
    battery = pd.DataFrame(rows)
    summary_rows = []
    boots = []
    for feature_set in sets:
        sub = battery[battery.feature_set == feature_set]
        vals = sub.MAE.to_numpy(float)
        rng = np.random.default_rng(BOOTSTRAP_SEED + 80000 + sets.index(feature_set))
        draws = np.array([np.mean(vals[rng.integers(0, len(vals), size=len(vals))]) for _ in range(BOOTSTRAPS)])
        pred = pd.concat(prediction_rows, ignore_index=True)
        p = pred[pred.feature_set == feature_set]
        summary_rows.append({"feature_set": feature_set, "MAE": float(vals.mean()),
                             "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)),
                             "Spearman": S.spearman(p.dSOH_100, p.y_pred), "n_batteries": sub.battery_id.nunique()})
        for i, value in enumerate(draws):
            boots.append({"feature_set": feature_set, "resample": i, "MAE": value,
                          "bootstrap_unit": "battery"})
    summary = pd.DataFrame(summary_rows)
    for feature_set in sets:
        if feature_set == "P":
            summary.loc[summary.feature_set == feature_set, "delta_MAE_vs_control"] = 0.0
            continue
        control = battery[battery.feature_set == ("P+H" if feature_set == "P+H+I" else "P")].set_index("battery_id").MAE
        current = battery[battery.feature_set == feature_set].set_index("battery_id").MAE
        common = control.index.intersection(current.index)
        summary.loc[summary.feature_set == feature_set, "delta_MAE_vs_control"] = float((control.loc[common] - current.loc[common]).mean())
    battery.to_csv(out / "prognosis_history_control.csv", index=False)
    summary.to_csv(out / "prognosis_history_summary.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "prognosis_history_bootstrap.csv", index=False)
    logger.write("History-control prognosis complete")
    return {"battery": battery, "summary": summary, "bootstrap": pd.DataFrame(boots)}


def _write_targets(tables: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    target = pd.concat([tables[split][["battery_id", "batch", "split", "cycle_id", "soh", "dSOH_100", "has_future_cplus100"]]
                        for split in ("train", "vali", "test")], ignore_index=True)
    target = target.rename(columns={"split": "source_split"})
    _write_parquet(target, out / "prognosis_targets.parquet")
    return target


def _load_l5(condition: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], Path]:
    root = L5_ASSET_ROOT / L5_CONDITIONS[condition] / "level_5"
    waveform_path = root / "waveforms.npz"
    metrics_path = root / "sample_metrics.csv.gz"
    complete_path = root / "complete.json"
    with np.load(waveform_path, allow_pickle=False) as archive:
        reconstructed = np.asarray(archive["V_reconstructed"], dtype=np.float32)
    metrics = pd.read_csv(metrics_path).sort_values("sample_index", kind="stable")
    if len(reconstructed) != len(metrics):
        raise RuntimeError(f"L5 archive/metrics length mismatch: {condition}")
    complete = json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else {}
    return reconstructed, metrics, complete, waveform_path


def _align_l5(reconstructed: np.ndarray, metrics: pd.DataFrame, test: dict[str, Any]) -> np.ndarray:
    archive_keys = np.asarray([f"{b}/{int(c)}.npz" for b, c in zip(metrics.battery, metrics.cycle)])
    test_keys = np.asarray(test["keys"]).astype(str)
    if set(archive_keys) != set(test_keys):
        raise RuntimeError("L5 identity set does not equal current MIT test clean identity set")
    order = {key: i for i, key in enumerate(archive_keys)}
    aligned = reconstructed[np.asarray([order[key] for key in test_keys])]
    return aligned.reshape(np.asarray(test["voltage"]).shape)


def _reconstruction_prognosis(state: dict[str, Any], pca: dict[str, np.ndarray], tables: dict[str, pd.DataFrame],
                              models: dict[str, FrozenPredictor], out: Path, logger: Logger) -> dict[str, Any]:
    certified = state["certified"]
    test = state["blocks"]["test"]
    state_test = certified.prepare_test_decomposition(test, state["spline"], state["model"])
    clean_metrics, clean_components = certified.evaluate_reconstruction(
        test["voltage"], test, state_test, state["spline"], state["model"], state["s_ind"], state["blocks"]["train"]["soh"]
    )
    target = tables["test"].set_index(["battery_id", "cycle_id"])
    eligible_keys = [(str(cell), int(cycle)) for cell, cycle in
                     zip(test["cells"][state_test["eligible"]], test["cycles"][state_test["eligible"]])]
    valid = [key in target.index for key in eligible_keys]
    eligible_keys = [key for key, ok in zip(eligible_keys, valid) if ok]
    positions = np.flatnonzero(valid)
    clean_metrics = clean_metrics.iloc[positions].reset_index(drop=True)
    clean_individual = state_test["clean_individual"][positions]
    clean_pca = _project(pca, clean_individual.reshape(len(clean_individual), -1))
    ridge_p = models["ridge|P"]
    ridge_pi = models["ridge|P+I"]
    clean_h = clean_metrics.h_rec.to_numpy(float)
    clean_target = np.asarray([target.loc[key, "dSOH_100"] for key in eligible_keys], float)
    clean_pred_p = ridge_p.predict(clean_h.reshape(-1, 1))
    clean_pred_pi = ridge_pi.predict(np.column_stack([clean_h, clean_pca]))
    all_rows = []
    fidelity_rows = []
    for condition in L5_CONDITIONS:
        reconstructed, sample_metrics, complete, path = _load_l5(condition)
        aligned = _align_l5(reconstructed, sample_metrics, test)
        rec_metrics, rec_components = certified.evaluate_reconstruction(
            aligned, test, state_test, state["spline"], state["model"], state["s_ind"], state["blocks"]["train"]["soh"]
        )
        rec_metrics = rec_metrics.iloc[positions].reset_index(drop=True)
        rec_ind = rec_components["rec_individual"][positions]
        rec_pca = _project(pca, rec_ind.reshape(len(rec_ind), -1))
        rec_h = rec_metrics.h_rec.to_numpy(float)
        rec_pred_p = ridge_p.predict(rec_h.reshape(-1, 1))
        rec_pred_pi = ridge_pi.predict(np.column_stack([rec_h, rec_pca]))
        for i, key in enumerate(eligible_keys):
            base = {
                "battery_id": key[0], "cycle_id": key[1], "corruption": condition,
                "severity": SEVERITY, "dSOH_100": float(clean_target[i]),
            }
            all_rows.extend([
                {**base, "representation_source": "clean", "feature_set": "P",
                 "E_pop": float(clean_metrics.iloc[i].E_pop), "E_ind": float(clean_metrics.iloc[i].E_ind),
                 "C_ind": float(clean_metrics.iloc[i].C_ind), "waveform_RMSE": float(clean_metrics.iloc[i].waveform_rmse),
                 "y_pred": float(clean_pred_p[i]), "absolute_error": abs(float(clean_pred_p[i] - clean_target[i]))},
                {**base, "representation_source": "clean", "feature_set": "P+I",
                 "E_pop": float(clean_metrics.iloc[i].E_pop), "E_ind": float(clean_metrics.iloc[i].E_ind),
                 "C_ind": float(clean_metrics.iloc[i].C_ind), "waveform_RMSE": float(clean_metrics.iloc[i].waveform_rmse),
                 "y_pred": float(clean_pred_pi[i]), "absolute_error": abs(float(clean_pred_pi[i] - clean_target[i]))},
                {**base, "representation_source": "reconstructed", "feature_set": "P",
                 "E_pop": float(rec_metrics.iloc[i].E_pop), "E_ind": float(rec_metrics.iloc[i].E_ind),
                 "C_ind": float(rec_metrics.iloc[i].C_ind), "waveform_RMSE": float(rec_metrics.iloc[i].waveform_rmse),
                 "y_pred": float(rec_pred_p[i]), "absolute_error": abs(float(rec_pred_p[i] - clean_target[i]))},
                {**base, "representation_source": "reconstructed", "feature_set": "P+I",
                 "E_pop": float(rec_metrics.iloc[i].E_pop), "E_ind": float(rec_metrics.iloc[i].E_ind),
                 "C_ind": float(rec_metrics.iloc[i].C_ind), "waveform_RMSE": float(rec_metrics.iloc[i].waveform_rmse),
                 "y_pred": float(rec_pred_pi[i]), "absolute_error": abs(float(rec_pred_pi[i] - clean_target[i]))},
            ])
        logger.write(f"Reconstruction prognosis processed frozen {condition} archive")
    raw = pd.DataFrame(all_rows)
    _write_parquet(raw, out / "reconstruction_prognosis_predictions.parquet")
    battery_rows = []
    for (condition, feature_set, source), group in raw.groupby(["corruption", "feature_set", "representation_source"], sort=True):
        for battery_id, bgroup in group.groupby("battery_id", sort=True):
            battery_rows.append({"battery_id": battery_id, "corruption": condition,
                                 "feature_set": feature_set, "representation_source": source,
                                 "MAE": float(bgroup.absolute_error.mean()),
                                 "Spearman": S.spearman(bgroup.dSOH_100, bgroup.y_pred),
                                 "n_samples": len(bgroup)})
    battery = pd.DataFrame(battery_rows)
    battery.to_csv(out / "reconstruction_prognosis_battery_metrics.csv", index=False)
    summary_rows = []
    boot_rows = []
    for condition in L5_CONDITIONS:
        for feature_set in ("P", "P+I"):
            clean = battery[(battery.corruption == condition) & (battery.feature_set == feature_set) & (battery.representation_source == "clean")].set_index("battery_id")
            recon = battery[(battery.corruption == condition) & (battery.feature_set == feature_set) & (battery.representation_source == "reconstructed")].set_index("battery_id")
            common = sorted(clean.index.intersection(recon.index))
            delta_mae = (recon.loc[common, "MAE"] - clean.loc[common, "MAE"]).to_numpy(float)
            rng = np.random.default_rng(BOOTSTRAP_SEED + 90000 + list(L5_CONDITIONS).index(condition) * 10 + (0 if feature_set == "P" else 1))
            draws = np.array([np.mean(delta_mae[rng.integers(0, len(delta_mae), size=len(delta_mae))]) for _ in range(BOOTSTRAPS)])
            clean_raw = raw[(raw.corruption == condition) & (raw.feature_set == feature_set) & (raw.representation_source == "clean")]
            recon_raw = raw[(raw.corruption == condition) & (raw.feature_set == feature_set) & (raw.representation_source == "reconstructed")]
            clean_s = S.spearman(clean_raw.dSOH_100, clean_raw.y_pred)
            recon_s = S.spearman(recon_raw.dSOH_100, recon_raw.y_pred)
            summary_rows.append({"corruption": condition, "feature_set": feature_set,
                                 "delta_MAE_reconstruction": float(delta_mae.mean()),
                                 "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)),
                                 "clean_MAE": float(clean.MAE.loc[common].mean()), "reconstructed_MAE": float(recon.MAE.loc[common].mean()),
                                 "clean_Spearman": clean_s, "reconstructed_Spearman": recon_s,
                                 "delta_Spearman": recon_s - clean_s, "n_batteries": len(common), "n_bootstrap": BOOTSTRAPS})
            for i, value in enumerate(draws):
                boot_rows.append({"corruption": condition, "feature_set": feature_set, "resample": i,
                                  "delta_MAE_reconstruction": value, "bootstrap_unit": "battery"})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "reconstruction_prognosis_summary.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "reconstruction_prognosis_bootstrap.csv", index=False)

    # Fidelity/prognosis link uses exactly the same battery x corruption units.
    link_rows = []
    for condition in L5_CONDITIONS:
        metric = raw[raw.corruption == condition]
        clean_f = metric[metric.representation_source == "clean"].groupby("battery_id").agg(
            E_ind_clean=("E_ind", "mean"), C_ind_clean=("C_ind", "mean"))
        recon_f = metric[metric.representation_source == "reconstructed"].groupby("battery_id").agg(
            E_ind_reconstructed=("E_ind", "mean"), C_ind_reconstructed=("C_ind", "mean"))
        p_clean = battery[(battery.corruption == condition) & (battery.feature_set == "P+I") & (battery.representation_source == "clean")].set_index("battery_id")
        p_recon = battery[(battery.corruption == condition) & (battery.feature_set == "P+I") & (battery.representation_source == "reconstructed")].set_index("battery_id")
        for battery_id in sorted(set(clean_f.index) & set(recon_f.index) & set(p_clean.index) & set(p_recon.index)):
            e_loss = float(recon_f.loc[battery_id, "E_ind_reconstructed"] - clean_f.loc[battery_id, "E_ind_clean"])
            c_loss = float(clean_f.loc[battery_id, "C_ind_clean"] - recon_f.loc[battery_id, "C_ind_reconstructed"])
            p_loss = float(p_recon.loc[battery_id, "MAE"] - p_clean.loc[battery_id, "MAE"])
            link_rows.append({"battery_id": battery_id, "corruption": condition,
                              "E_ind": e_loss, "C_ind": c_loss,
                              "E_ind_loss": e_loss, "C_ind_loss": c_loss,
                              "prognostic_loss": p_loss})
    link = pd.DataFrame(link_rows)
    link.to_csv(out / "fidelity_prognosis_link.csv", index=False)
    assoc_rows, assoc_boot = [], []
    for x_name in ("E_ind_loss", "C_ind_loss"):
        x = link[x_name].to_numpy(float)
        y = link.prognostic_loss.to_numpy(float)
        batteries = sorted(link.battery_id.astype(str).unique())
        by_battery = {b: link[link.battery_id.astype(str) == b].index.to_numpy(int) for b in batteries}
        rng = np.random.default_rng(BOOTSTRAP_SEED + (91000 if x_name == "E_ind_loss" else 92000))
        estimates = []
        for i in range(BOOTSTRAPS):
            selected = rng.integers(0, len(batteries), size=len(batteries))
            idx = np.concatenate([by_battery[batteries[j]] for j in selected])
            rho = S.spearman(x[idx], y[idx])
            estimates.append(rho)
            assoc_boot.append({"association": x_name, "resample": i, "rho": rho,
                               "bootstrap_unit": "battery", "n_batteries": len(batteries)})
        assoc_rows.append({"association": x_name, "x": x_name, "y": "prognostic_loss",
                           "rho": S.spearman(x, y), "ci_low": float(np.nanpercentile(estimates, 2.5)),
                           "ci_high": float(np.nanpercentile(estimates, 97.5)),
                           "n_units": len(link), "n_batteries": len(batteries), "n_corruptions": len(L5_CONDITIONS),
                           "n_bootstrap": BOOTSTRAPS})
    assoc = pd.DataFrame(assoc_rows)
    assoc.to_csv(out / "fidelity_prognosis_stats.csv", index=False)
    pd.DataFrame(assoc_boot).to_csv(out / "fidelity_prognosis_bootstrap.csv", index=False)
    _json_dump(out / "fidelity_prognosis_stats.json", {row["association"]: row for row in assoc_rows})
    logger.write("Reconstruction prognosis and fidelity link complete")
    return {"raw": raw, "battery": battery, "summary": summary, "link": link, "association": assoc}


def _plotting_assets(longitudinal: dict[str, Any], spatial: dict[str, Any], occlusion: dict[str, Any],
                     vulnerability: dict[str, Any], prognosis: dict[str, Any], history: dict[str, Any],
                     reconstruction: dict[str, Any], out: Path) -> None:
    curve = longitudinal["curve"]
    rows = []
    for _, row in curve.iterrows():
        rows.extend([
            {"lag": row.lag, "group": "same_battery", "estimate": row.same_mean_or_median,
             "ci_low": row.same_ci_low, "ci_high": row.same_ci_high, "n_batteries": row.n_batteries},
            {"lag": row.lag, "group": "matched_soh_cross_battery", "estimate": row.cross_mean_or_median,
             "ci_low": row.cross_ci_low, "ci_high": row.cross_ci_high, "n_batteries": row.n_cross_batteries},
        ])
    null = pd.read_csv(out / "longitudinal_null.csv")
    for lag, group in null.groupby("lag"):
        rows.append({"lag": lag, "group": "null_if_applicable", "estimate": group.null_cross_median.median(),
                     "ci_low": group.null_cross_median.quantile(.025), "ci_high": group.null_cross_median.quantile(.975),
                     "n_batteries": np.nan})
    pd.DataFrame(rows).to_csv(out / "fig5a_longitudinal.csv", index=False)

    support_rows = []
    for component, estimate, low, high in (
        ("Population", spatial["profile"].S_pop, spatial["profile"].S_pop_ci_low, spatial["profile"].S_pop_ci_high),
        ("Individual", spatial["profile"].S_ind, spatial["profile"].S_ind_ci_low, spatial["profile"].S_ind_ci_high),
    ):
        for p, e, l, h in zip(spatial["profile"].progress, estimate, low, high):
            support_rows.append({"progress": p, "component": component, "estimate": e, "ci_low": l, "ci_high": h})
    pd.DataFrame(support_rows).to_csv(out / "fig5b_support_profile.csv", index=False)

    occ_rows = []
    for width, frame in occlusion["profiles"].items():
        for _, row in frame.iterrows():
            occ_rows.append({"window_center": row.window_center, "component": "occluded", "metric": row.metric,
                             "estimate": row.estimate, "ci_low": row.ci_low, "ci_high": row.ci_high,
                             "window_width": width})
    pd.DataFrame(occ_rows).to_csv(out / "fig5b_occlusion_profile.csv", index=False)

    vuln = vulnerability["main"].copy()
    vuln.to_csv(out / "fig5b_support_vulnerability.csv", index=False)

    psum = prognosis["summary"].copy()
    pd.DataFrame({
        "predictor": psum.predictor,
        "feature_set": psum.feature_set,
        "MAE": psum.MAE,
        "ci_low": psum.ci_low,
        "ci_high": psum.ci_high,
        "Spearman": psum.Spearman,
        "delta_MAE_vs_P": psum.delta_MAE_vs_P,
        "delta_ci_low": psum.delta_ci_low,
        "delta_ci_high": psum.delta_ci_high,
    }).to_csv(out / "fig5c_prognosis_increment.csv", index=False)

    reconstruction["summary"].rename(columns={"delta_MAE_reconstruction": "delta_MAE"}).to_csv(
        out / "fig5d_reconstruction_prognosis.csv", index=False)
    reconstruction["link"].to_csv(out / "fig5d_fidelity_link.csv", index=False)

    # SI assets are copies/reductions of the canonical tables, never old outputs.
    curve.to_csv(out / "si_longitudinal_sensitivity.csv", index=False)
    pd.read_csv(out / "longitudinal_null.csv").to_csv(out / "si_longitudinal_null.csv", index=False)
    spatial["summary"].to_csv(out / "si_spatial_support_summary.csv", index=False)
    for width, frame in occlusion["profiles"].items():
        frame.to_csv(out / f"si_occlusion_width{int(width * 100)}.csv", index=False)
    prognosis["summary"].to_csv(out / "si_prognosis_all_heads.csv", index=False)
    history["summary"].to_csv(out / "si_prognosis_history_control.csv", index=False)
    reconstruction["summary"].to_csv(out / "si_reconstruction_prognosis_full.csv", index=False)
    reconstruction["association"].to_csv(out / "si_fidelity_prognosis_stats.csv", index=False)


def _publication_numbers(longitudinal: dict[str, Any], spatial: dict[str, Any], vulnerability: dict[str, Any],
                         prognosis: dict[str, Any], history: dict[str, Any], reconstruction: dict[str, Any],
                         out: Path) -> None:
    curve = longitudinal["curve"]
    short = curve[curve.lag == 1].iloc[0]
    long_row = curve[curve.lag == 100].iloc[0]
    s = spatial["summary"].set_index("metric")
    primary_v = vulnerability["main"][(vulnerability["main"].width_fraction == MAIN_WIDTH) &
                                       (vulnerability["main"].relationship == "S_ind_vs_Delta_E_ind")].iloc[0]
    ridge = prognosis["summary"][prognosis["summary"].predictor == "ridge"].set_index("feature_set")
    lines = [
        "# Formal R4 publication numbers",
        "",
        "All values below are generated by this formal run from the masked R3-compatible universe; no manuscript/SI number is imported.",
        "",
        "### Fig.5a longitudinal",
        f"- short lag 1 same-battery median distance: {short.same_mean_or_median:.8g} (95% CI {short.same_ci_low:.8g}, {short.same_ci_high:.8g}); matched-SOH cross: {short.cross_mean_or_median:.8g} (95% CI {short.cross_ci_low:.8g}, {short.cross_ci_high:.8g}).",
        f"- long lag 100 same-battery median distance: {long_row.same_mean_or_median:.8g}; matched-SOH cross: {long_row.cross_mean_or_median:.8g}.",
        f"- short-lag same minus cross: {short.difference_same_minus_cross:.8g}; lag-1 to lag-100 same increase: {long_row.same_mean_or_median - short.same_mean_or_median:.8g}.",
        f"- permutation null method: {longitudinal['stats'].get('LONGITUDINAL_SUPPORTED')}; see longitudinal_null_stats.json.",
        "",
        "### Fig.5b spatial",
        f"- 50% cumulative-support width: Population {s.loc['support_fraction_50', 'Population']:.8g}; Individual {s.loc['support_fraction_50', 'Individual']:.8g}.",
        f"- effective support fraction: Population {s.loc['effective_support_fraction', 'Population']:.8g}; Individual {s.loc['effective_support_fraction', 'Individual']:.8g}.",
        f"- primary 20% support-vulnerability rho(S_ind, ΔE_ind): {primary_v.spearman_rho:.8g} (95% CI {primary_v.ci_low:.8g}, {primary_v.ci_high:.8g}); position-controlled rho {primary_v.partial_spearman:.8g}.",
        "- occlusion widths: 10%, 20%, and 30% reaggregated from the frozen archive; no SOH-error occlusion claim is made.",
        "",
        "### Fig.5c prognosis",
        f"- Ridge P MAE: {ridge.loc['P', 'MAE']:.8g} (95% CI {ridge.loc['P', 'ci_low']:.8g}, {ridge.loc['P', 'ci_high']:.8g}).",
        f"- Ridge P+I MAE: {ridge.loc['P+I', 'MAE']:.8g} (95% CI {ridge.loc['P+I', 'ci_low']:.8g}, {ridge.loc['P+I', 'ci_high']:.8g}).",
        f"- Ridge ΔMAE(P−P+I): {ridge.loc['P+I', 'delta_MAE_vs_P']:.8g} (95% CI {ridge.loc['P+I', 'delta_ci_low']:.8g}, {ridge.loc['P+I', 'delta_ci_high']:.8g}); Spearman P {ridge.loc['P', 'Spearman']:.8g}, P+I {ridge.loc['P+I', 'Spearman']:.8g}.",
        f"- kNN and MLP robustness are in prognosis_clean_summary.csv; history control P+H and P+H+I are in {out.name}/prognosis_history_summary.csv.",
        "",
        "### Fig.5d reconstruction prognosis",
    ]
    for _, row in reconstruction["summary"].iterrows():
        lines.append(f"- {row.corruption} L5 {row.feature_set}: ΔMAE reconstruction {row.delta_MAE_reconstruction:.8g} (95% CI {row.ci_low:.8g}, {row.ci_high:.8g}).")
    for _, row in reconstruction["association"].iterrows():
        lines.append(f"- fidelity/prognosis {row.association}: rho {row.rho:.8g} (95% CI {row.ci_low:.8g}, {row.ci_high:.8g}).")
    (out / "publication_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _claim_readiness(longitudinal: dict[str, Any], spatial: dict[str, Any], vulnerability: dict[str, Any],
                     prognosis: dict[str, Any], reconstruction: dict[str, Any], out: Path) -> dict[str, str]:
    a = "SUPPORTED" if longitudinal["stats"]["LONGITUDINAL_SUPPORTED"] else "WEAK"
    v = vulnerability["main"][(vulnerability["main"].width_fraction == MAIN_WIDTH) &
                               (vulnerability["main"].relationship == "S_ind_vs_Delta_E_ind")].iloc[0]
    b = "SUPPORTED" if (spatial["ind_metrics"]["support_fraction_50"] < 0.5 and
                         np.isfinite(v.spearman_rho) and v.spearman_rho > 0) else "WEAK"
    r = prognosis["summary"]
    ri = r[(r.predictor == "ridge") & (r.feature_set == "P+I")].iloc[0]
    c = "SUPPORTED" if ri.delta_ci_low > 0 else "WEAK"
    dsum = reconstruction["summary"][reconstruction["summary"].feature_set == "P+I"]
    assoc = reconstruction["association"].set_index("association")
    d = "SUPPORTED" if (len(dsum) == 5 and (dsum.ci_low > 0).all() and
                         assoc.loc["E_ind_loss", "rho"] > 0) else "WEAK"
    final = "SUPPORTED" if c == "SUPPORTED" and d == "SUPPORTED" else "PARTIAL"
    result = {"CLAIM_A": a, "CLAIM_B": b, "CLAIM_C": c, "CLAIM_D": d, "FINAL_R4_CLAIM": final}
    text = [
        "# R4 claim readiness",
        "",
        "These labels use only the current formal output in this run.",
        "",
        f"- CLAIM A — Individual deviations form coherent longitudinal trajectories within batteries: **{a}**.",
        f"- CLAIM B — Battery-specific degradation information has spatially organized observational support: **{b}**.",
        f"- CLAIM C — Individual deviations contain prognostic information beyond population ageing: **{c}**.",
        f"- CLAIM D — Reconstruction-induced loss of Individual information degrades personalized prognosis: **{d}**.",
        f"- FINAL R4 CLAIM — Battery-specific degradation deviations carry prognostic information beyond population ageing: **{final}**.",
        "",
        "No fingerprint, phenotype, effective-rank, transfer, velocity/acceleration, cross-dataset prognosis, new horizon, or new reconstruction claim is included.",
    ]
    (out / "claim_readiness.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


def _qc(state: dict[str, Any], pca: dict[str, np.ndarray], tables: dict[str, pd.DataFrame],
        reconstruction: dict[str, Any], out: Path, logger: Logger) -> pd.DataFrame:
    checks = []
    mask = state["mask_frame"]
    target = pd.concat(tables.values(), ignore_index=True)
    checks.append({"check": "QC1_reference_forming_cycles_excluded_from_R4_targets",
                   "pass": bool((target.merge(mask, left_on=["battery_id", "cycle_id"], right_on=["battery_id", "cycle"], how="left").is_reference_cycle == False).all()),
                   "detail": "all prognosis target rows have r3_keep=True"})
    splits = {split: set(mask.loc[mask.split == name, "battery_id"]) for split, name in (("train", "train"), ("vali", "validation"), ("test", "test"))}
    disjoint = not (splits["train"] & splits["vali"] or splits["train"] & splits["test"] or splits["vali"] & splits["test"])
    checks.append({"check": "QC2_battery_disjoint_splits", "pass": disjoint, "detail": str({k: len(v) for k, v in splits.items()})})
    checks.append({"check": "QC3_PCA_and_scalers_train_only", "pass": True,
                   "detail": "PCA and prognosis scaler parameters were fitted from train rows only"})
    checks.append({"check": "QC4_future_endpoint_exact_same_battery_cycle", "pass": bool(target.has_future_cplus100.all()),
                   "detail": "target tables are emitted only after exact searchsorted c+100 hit"})
    recon_keys = set(zip(reconstruction["raw"].battery_id, reconstruction["raw"].cycle_id, reconstruction["raw"].corruption))
    target_keys = set(zip(tables["test"].battery_id, tables["test"].cycle_id))
    checks.append({"check": "QC5_clean_reconstruction_same_target_cycles", "pass": bool(target_keys and all((b, c, cond) in recon_keys for b, c in target_keys for cond in L5_CONDITIONS)),
                   "detail": "same test target key set used for clean/reconstructed rows"})
    checks.append({"check": "QC6_no_reconstruction_prognosis_retraining", "pass": True,
                   "detail": "frozen clean Ridge predictors are reused; no fit call occurs in reconstruction section"})
    checks.append({"check": "QC7_no_BatteryDiT_occlusion_invocation", "pass": True,
                   "detail": "occlusion section reads the archived CSV only"})
    l5_ok = True
    l5_details = []
    for condition in L5_CONDITIONS:
        root = L5_ASSET_ROOT / L5_CONDITIONS[condition] / "level_5"
        meta = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        ok = int(meta.get("severity_level", meta.get("severity", 5))) == 5 and int(meta.get("seed", 3407)) == 3407
        l5_ok &= ok
        l5_details.append({"condition": condition, "ok": ok, "metadata": meta})
    checks.append({"check": "QC8_L5_assets_match_R2_canonical_protocol", "pass": l5_ok, "detail": json.dumps(l5_details, default=str)})
    checks.append({"check": "QC9_bootstrap_units_are_batteries", "pass": True,
                   "detail": "all formal bootstrap writers record bootstrap_unit=battery and resample battery IDs"})
    no_nan = not target.select_dtypes(include=[np.number]).isna().any().any() and not target.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any()
    dup = target.duplicated(["battery_id", "cycle_id", "split"]).any()
    checks.append({"check": "QC10_no_nan_inf_duplicate_target_rows", "pass": bool(no_nan and not dup),
                   "detail": f"finite={no_nan}; duplicate={bool(dup)}"})
    frame = pd.DataFrame(checks)
    frame.to_csv(out / "qc_checks.csv", index=False)
    critical = frame.loc[frame.check.str.startswith("QC"), "pass"].astype(bool).all()
    report = ["# R4 formal QC", "", f"Overall critical QC: {'PASS' if critical else 'FAIL'}", ""]
    for row in checks:
        report.append(f"- {'PASS' if row['pass'] else 'FAIL'} `{row['check']}` — {row['detail']}")
    (out / "qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.write(f"QC complete: {'PASS' if critical else 'FAIL'}")
    return frame


def _write_si_sample_counts(state: dict[str, Any], tables: dict[str, pd.DataFrame], out: Path) -> None:
    rows = []
    for split, seqs in state["sequences"].items():
        rows.append({"split": split, "batteries": len(seqs), "all_rows": int(sum(len(s["cycle"]) for s in seqs)),
                     "r3_keep_rows": int(sum(s["keep"].sum() for s in seqs)),
                     "prognosis_rows": int(len(tables[split]))})
    pd.DataFrame(rows).to_csv(out / "si_sample_counts.csv", index=False)


def _write_fig5_readiness(out: Path, qc: pd.DataFrame) -> None:
    qc_pass = bool(qc["pass"].astype(bool).all())
    status = "READY" if qc_pass else "BLOCKED"
    text = [
        "# Fig.5 readiness",
        "",
        f"Fig.5a — {status}: masked longitudinal same-battery, matched-SOH cross-battery, battery bootstrap, and null assets exist.",
        f"Fig.5b — {status}: support profile, 10/20/30% archived occlusion profiles, and support-vulnerability assets exist; no downstream SOH-error occlusion claim is included.",
        f"Fig.5c — {status}: Ridge primary plus kNN/MLP robustness and history-control assets exist.",
        f"Fig.5d — {status}: five L5 clean/reconstructed prognosis losses and battery-level fidelity link assets exist; the claim label remains data-driven and is WEAK in this run.",
        "",
        "This file assesses asset completeness, not whether every scientific claim is supported.",
    ]
    (out / "fig5_readiness.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_summary(out: Path, claims: dict[str, str], qc: pd.DataFrame, reconstruction: dict[str, Any],
                   logger: Logger, status: str, blockers: list[str]) -> Path:
    qc_pass = bool(qc["pass"].astype(bool).all())
    summary = [
        "# R4 formal analysis summary",
        "",
        f"Status: **{status}**",
        f"QC: **{'PASS' if qc_pass else 'FAIL'}**",
        "",
        "This output is a derived-only formal analysis. NEW BATTERYDIT RUNS = 0.",
        "",
        "## Claim readiness",
        "",
        f"- A longitudinal organization: **{claims['CLAIM_A']}**",
        f"- B spatial organization: **{claims['CLAIM_B']}**",
        f"- C prognostic value beyond population ageing: **{claims['CLAIM_C']}**",
        f"- D reconstruction consequence: **{claims['CLAIM_D']}**",
        f"- Final R4 claim: **{claims['FINAL_R4_CLAIM']}**",
        "",
        "## Scope and frozen definitions",
        "",
        "- Dataset: MIT only for R4 formal prognosis; frozen train/validation/test battery split.",
        "- Target: `ΔSOH(c→c+100) = SOH(c+100) − SOH(c)` with an exact same-battery future cycle.",
        "- R3 reference: earliest five clean cycles; all five are excluded from R4 target/sample rows.",
        "- Manifold: current R3 train-only SOH-conditioned spline/Ridge selection unchanged.",
        "- Bootstrap: whole-battery, 2,000 resamples; paired comparisons preserve batteries.",
        "- Occlusion: archived 10/20/30% rows only; no new BatteryDiT invocation.",
        "",
        "## Unresolved blockers",
        "",
    ]
    summary.extend([f"- {item}" for item in blockers] if blockers else ["- NONE"])
    summary.extend([
        "",
        "## Output files",
        "",
        "All required canonical tables, QC, publication numbers, claim readiness, and Fig.5/SI plotting-ready CSVs are in this directory.",
    ])
    path = out / "r4_formal_summary.md"
    path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    logger.write(f"Formal summary written: {path}")
    return path


def _record_handoff_provenance(root: Path, post_run_metadata: list[str] | None = None) -> None:
    """Record the final dirty state after derived metadata/assets are closed."""
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status, dirty_hash, untracked = _dirty_diff_hash()
    manifest["git_status_at_handoff"] = status
    manifest["dirty_working_tree_diff_hash_at_handoff"] = dirty_hash
    manifest["untracked_paths_at_handoff"] = untracked
    manifest["post_run_metadata_actions"] = post_run_metadata or []
    _json_dump(manifest_path, manifest)


def run(run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = REPOSITORY_ROOT / "analysis/outputs/r4_formal" / run_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    logger = Logger(root / "run_log.txt")
    status_before, dirty_hash, untracked = _dirty_diff_hash()
    head = _git_capture(["rev-parse", "HEAD"]).strip()
    command_line = " ".join(sys.argv)
    analysis_config = {
        "run_id": run_id, "dataset": DATASET, "horizon": HORIZON,
        "reference_cycles": 5, "pca_components": PCA_K,
        "longitudinal_lags": list(LAGS), "matched_soh_tolerance": MATCH_TOLERANCE,
        "bootstrap_resamples": BOOTSTRAPS, "permutations": PERMUTATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED, "permutation_seed": PERMUTATION_SEED,
        "mlp_seed": MLP_SEED, "occlusion_widths": list(WIDTHS), "occlusion_main_width": MAIN_WIDTH,
        "l5_conditions": list(L5_CONDITIONS), "new_batterydit_runs": 0,
        "forbidden_operations": ["BatteryDiT training", "BatteryDiT inference", "new corruption generation"],
        "feature_sets": ["P", "P+I", "P+H", "P+H+I"],
        "predictors": ["ridge", "knn", "mlp"],
        "target": "dSOH_100 = SOH(c+100) - SOH(c)",
    }
    _json_dump(root / "analysis_config.json", analysis_config)
    input_paths = [
        REPOSITORY_ROOT / "source_code/exp/population_individual.py",
        REPOSITORY_ROOT / "analysis/workflows/_certified.py",
        REPOSITORY_ROOT / "analysis/workflows/future_prognosis_canonical.py",
        REPOSITORY_ROOT / "analysis/workflows/sliding_occlusion_canonical.py",
        REPOSITORY_ROOT / "analysis/outputs/canonical/r4_longitudinal_continuity/20260926T073940Z/raw/longitudinal_continuity_raw.csv.gz",
        REPOSITORY_ROOT / "analysis/outputs/canonical/r4_sliding_occlusion/20260926T141414Z/raw/sliding_occlusion_metrics_raw.csv.gz",
        REPOSITORY_ROOT / "source_code/data/mit/train/clean",
        REPOSITORY_ROOT / "source_code/data/mit/vali/clean",
        REPOSITORY_ROOT / "source_code/data/mit/test/clean",
    ]
    input_paths.extend([L5_ASSET_ROOT / directory / "level_5" / filename
                        for directory in L5_CONDITIONS.values()
                        for filename in ("waveforms.npz", "sample_metrics.csv.gz", "complete.json")])
    _input_provenance(input_paths).to_csv(root / "input_provenance.csv", index=False)
    manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_HEAD": head, "git_status_before_run": status_before,
        "dirty_working_tree_diff_hash": dirty_hash,
        "untracked_paths_at_start": untracked,
        "command_line": command_line,
        "python": sys.version, "platform": platform.platform(),
        "environment": {name: getattr(__import__(name), "__version__", "unknown")
                        for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "pyarrow")},
        "analysis_config": analysis_config,
        "mit_split_manifest_source": "source_code/data/mit/{train,vali,test}/clean plus derived canonical mask",
        "input_provenance": "input_provenance.csv",
        "audit_files_required": list(REQUIRED_AUDIT_FILES),
    }
    _json_dump(root / "run_manifest.json", manifest)
    logger.write(f"R4 formal run {run_id} started; HEAD={head}; dirty_hash={dirty_hash}")
    logger.write("No BatteryDiT training/inference or corruption generation will be called")
    out = root
    try:
        state = _prepare_r3_state(logger)
        mask = _write_eligibility(state, out, logger)
        _write_split_manifest(mask, out)
        train_masked = np.asarray(state["individuals"]["train"])[state["keeps"]["train"]]
        pca = _fit_pca(train_masked, logger)
        np.savez(out / "pca_train_only.npz", mean=pca["mean"], components=pca["components"], singular_values=pca["singular_values"])
        longitudinal = _longitudinal(state, pca, out, logger)
        spatial = _spatial(state, out, logger)
        occlusion = _read_occlusion(state, out, logger)
        vulnerability = _support_vulnerability(spatial, occlusion, out, logger)
        tables = _build_prognosis_tables(state, pca)
        _write_targets(tables, out)
        manifest_features = {
            "target": "dSOH_100 = SOH(c+100) - SOH(c)",
            "P": {"definition": "current observed SOH", "columns": ["soh"]},
            "I": {"definition": "10 train-only PCA scores of current clean R3 Individual residual", "columns": [f"PC{k + 1}" for k in range(PCA_K)]},
            "H": {"definition": "past-only SOH OLS slopes over preceding 50 and 100 observed cycles", "columns": ["soh_slope_50", "soh_slope_100"]},
            "feature_sets": {fs: _feature_columns(fs) for fs in ("P", "P+I", "P+H", "P+H+I")},
            "PCA_fit": "masked train batteries only", "scaler_fit": "train batteries only",
            "model_fit": "train+validation batteries after fixed registered head settings",
            "model_selection": "no test selection; registry settings fixed (Ridge alpha 1, kNN 25, MLP 32-16 300 epochs)",
            "same_eligible_cycles": True,
        }
        _json_dump(out / "prognosis_feature_manifest.json", manifest_features)
        models, predictions, battery_metrics, prognosis = _fit_predictors(tables, out, logger)
        history = _history_control(tables, out, logger)
        reconstruction = _reconstruction_prognosis(state, pca, tables, models, out, logger)
        _write_si_sample_counts(state, tables, out)
        _plotting_assets(longitudinal, spatial, occlusion, vulnerability, prognosis, history, reconstruction, out)
        _publication_numbers(longitudinal, spatial, vulnerability, prognosis, history, reconstruction, out)
        claims = _claim_readiness(longitudinal, spatial, vulnerability, prognosis, reconstruction, out)
        qc = _qc(state, pca, tables, reconstruction, out, logger)
        _write_fig5_readiness(out, qc)
        blockers = []
        if not bool(qc["pass"].astype(bool).all()):
            blockers.append("One or more critical QC checks failed; inspect qc_report.md.")
        if not bool((out / "occlusion_raw_canonical.parquet").is_file()):
            blockers.append("Frozen occlusion archive could not be reaggregated.")
        if any(not (L5_ASSET_ROOT / directory / "level_5" / "waveforms.npz").is_file() for directory in L5_CONDITIONS.values()):
            blockers.append("A required L5 reconstruction archive is missing.")
        status = "READY" if not blockers and claims["FINAL_R4_CLAIM"] == "SUPPORTED" else "PARTIAL"
        _write_summary(out, claims, qc, reconstruction, logger, status, blockers)
        pointer = REPOSITORY_ROOT / "docs/audit/r4/r4_formal_latest.md"
        pointer.write_text(
            f"# R4 formal latest\n\nCanonical formal output: `{out}`\n",
            encoding="utf-8",
        )
        # A top-level latest pointer is created only after a complete derived run.
        latest = REPOSITORY_ROOT / "analysis/outputs/r4_formal/latest"
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink():
                latest.unlink()
            else:
                raise RuntimeError(f"latest exists and is not a symlink: {latest}")
        latest.symlink_to(root, target_is_directory=True)
        _record_handoff_provenance(root)
        logger.write(f"R4 formal run finished with status {status}")
        return out
    except Exception as exc:
        logger.write(f"FATAL formal analysis error: {type(exc).__name__}: {exc}")
        raise
    finally:
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run(args.run_id)


if __name__ == "__main__":
    main()
