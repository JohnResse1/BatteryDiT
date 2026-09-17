"""Prespecified physical degradation signatures and clean-only selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .diagnostic_features import delta_Q


POSITIONS = (0.20, 0.40, 0.60, 0.80)
SLOPE_HALF_WIDTH = 10
MIN_SELECTION_RHO = 0.10
MIN_EXTRACTION_SUCCESS = 0.80


def local_slope(curves: np.ndarray, center: int) -> np.ndarray:
    left = max(0, center - SLOPE_HALF_WIDTH)
    right = min(curves.shape[1], center + SLOPE_HALF_WIDTH + 1)
    progress = np.arange(left, right, dtype=np.float64) / (curves.shape[1] - 1)
    centered = progress - progress.mean()
    return np.asarray(curves[:, left:right], dtype=np.float64) @ centered / float(np.sum(centered ** 2))


def candidate_features(curves: np.ndarray, v_low: float, v_high: float) -> dict[str, np.ndarray]:
    curves = np.asarray(curves, dtype=np.float32).reshape(len(curves), -1)
    features: dict[str, np.ndarray] = {}
    for position in POSITIONS:
        percentage = int(position * 100)
        center = int(round(position * (curves.shape[1] - 1)))
        features[f"voltage_level_p{percentage}"] = np.asarray(curves[:, center], dtype=np.float64)
        features[f"local_slope_p{percentage}"] = local_slope(curves, center)
    features["voltage_window_capacity"] = delta_Q(curves, V_low=v_low, V_high=v_high)[:, 0]
    return features


def _rho(values: np.ndarray, soh: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(soh)
    if keep.sum() < 3 or np.unique(values[keep]).size < 2 or np.unique(soh[keep]).size < 2:
        return float("nan")
    return float(spearmanr(values[keep], soh[keep]).statistic)


def _stability(values: np.ndarray, cells: np.ndarray, cycles: np.ndarray) -> float:
    scale = float(np.subtract(*np.nanpercentile(values, (75, 25))))
    if scale <= 1e-12:
        return float("inf")
    changes = []
    for cell in np.unique(cells.astype(str)):
        indices = np.flatnonzero(cells == cell)
        indices = indices[np.argsort(cycles[indices], kind="stable")]
        finite = values[indices][np.isfinite(values[indices])]
        if len(finite) >= 2:
            changes.append(float(np.median(np.abs(np.diff(finite))) / scale))
    return float(np.median(changes)) if changes else float("nan")


def select_signatures(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]):
    """Select at most one signature per family using clean train/validation only."""
    train_voltage = np.asarray(train["voltage"], dtype=np.float32).reshape(len(train["voltage"]), -1)
    validation_voltage = np.asarray(validation["voltage"], dtype=np.float32).reshape(len(validation["voltage"]), -1)
    v_low, v_high = np.percentile(train_voltage, (10, 90)).astype(float)
    train_features = candidate_features(train_voltage, float(v_low), float(v_high))
    validation_features = candidate_features(validation_voltage, float(v_low), float(v_high))
    rows = []
    for name in train_features:
        family = "local_voltage" if name.startswith("voltage_level") else "local_slope" if name.startswith("local_slope") else "global_voltage_window"
        train_values, validation_values = train_features[name], validation_features[name]
        train_success = float(np.isfinite(train_values).mean())
        validation_success = float(np.isfinite(validation_values).mean())
        train_rho = _rho(train_values, train["soh"])
        validation_rho = _rho(validation_values, validation["soh"])
        finite = np.isfinite(train_rho) and np.isfinite(validation_rho)
        score = min(abs(train_rho), abs(validation_rho)) if finite else -np.inf
        eligible = bool(
            min(train_success, validation_success) >= MIN_EXTRACTION_SUCCESS
            and score >= MIN_SELECTION_RHO
            and np.sign(train_rho) == np.sign(validation_rho)
        )
        rows.append({
            "signature": name, "feature_family": family,
            "clean_train_rho": train_rho, "clean_validation_rho": validation_rho,
            "extraction_success_rate": min(train_success, validation_success),
            "train_cycle_stability": _stability(train_values, train["cells"], train["cycles"]),
            "validation_cycle_stability": _stability(validation_values, validation["cells"], validation["cycles"]),
            "selection_score_min_abs_rho": score,
            "selection_eligible": eligible, "selected": False,
        })
    table = pd.DataFrame(rows)
    selected: list[str] = []
    for family in ("local_voltage", "local_slope", "global_voltage_window"):
        eligible = table[(table.feature_family == family) & table.selection_eligible]
        if not eligible.empty:
            selected.append(str(eligible.sort_values("selection_score_min_abs_rho", ascending=False, kind="stable").iloc[0].signature))
    if len(selected) < 3:
        remaining = table[table.selection_eligible & ~table.signature.isin(selected)].sort_values("selection_score_min_abs_rho", ascending=False, kind="stable")
        selected.extend(remaining.signature.astype(str).tolist()[:3 - len(selected)])
    selected = selected[:3]
    if len(selected) != 3:
        raise RuntimeError(f"only {len(selected)} degradation signatures passed the clean train/validation gate")
    table.loc[table.signature.isin(selected), "selected"] = True
    scales = {name: float(np.subtract(*np.percentile(train_features[name], (75, 25)))) for name in selected}
    if any(not np.isfinite(value) or value <= 1e-12 for value in scales.values()):
        raise RuntimeError(f"invalid training IQR for selected degradation signatures: {scales}")
    return table, selected, scales, (float(v_low), float(v_high))


def signature_error(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    selected: list[str],
    scales: dict[str, float],
    voltage_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return E_sig: median IQR-normalized rec-versus-clean feature error."""
    clean_features = candidate_features(clean, *voltage_bounds)
    reconstructed_features = candidate_features(reconstructed, *voltage_bounds)
    per_signature = {
        name: np.abs(reconstructed_features[name] - clean_features[name]) / scales[name]
        for name in selected
    }
    values = np.column_stack(list(per_signature.values()))
    return np.nanmedian(values, axis=1), per_signature
