"""Shared battery-wise statistics for the result workflows.

Every uncertainty interval resamples whole batteries, never cycles.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return float("nan")
    xr = rankdata(x) - rankdata(x).mean()
    yr = rankdata(y) - rankdata(y).mean()
    denom = float(np.sqrt((xr ** 2).sum() * (yr ** 2).sum()))
    return float((xr * yr).sum() / denom) if denom > 0 else float("nan")


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]
    if len(y_true) < 2:
        return float("nan")
    ss = float(((y_true - y_true.mean()) ** 2).sum())
    return float(1 - ((y_true - y_pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.abs(y_true[keep] - y_pred[keep]).mean()) if keep.any() else float("nan")


def rank_accuracy(true, pred) -> float:
    """Fraction of correctly ordered pairs."""
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    agree = total = 0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            if true[i] == true[j]:
                continue
            total += 1
            if (true[i] < true[j]) == (pred[i] < pred[j]):
                agree += 1
    return agree / total if total else float("nan")


def battery_bootstrap(values: np.ndarray, batteries: np.ndarray, statistic,
                      n: int = 2000, seed: int = 20260926) -> tuple[float, float, float]:
    """Whole-battery bootstrap of an arbitrary statistic."""
    values = np.asarray(values, float)
    batteries = np.asarray(batteries)
    uniq = np.array(sorted(set(batteries)))
    index = {b: np.where(batteries == b)[0] for b in uniq}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[b] for b in draw])
        value = statistic(values[idx])
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan"), float("nan")
    observed = statistic(values)
    return float(observed), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def battery_bootstrap_pair(x, y, batteries, n: int = 2000, seed: int = 20260926):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    batteries = np.asarray(batteries)
    uniq = np.array(sorted(set(batteries)))
    index = {b: np.where(batteries == b)[0] for b in uniq}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[b] for b in draw])
        value = spearman(x[idx], y[idx])
        if np.isfinite(value):
            estimates.append(value)
    low, high = (np.percentile(estimates, (2.5, 97.5)) if estimates else (np.nan, np.nan))
    return float(spearman(x, y)), float(low), float(high)


def battery_bootstrap_delta_mae(y_true, pred_a, pred_b, batteries,
                                n: int = 2000, seed: int = 20260926):
    """CI for MAE_b - MAE_a (positive => model a is better)."""
    y_true = np.asarray(y_true, float)
    pred_a = np.asarray(pred_a, float)
    pred_b = np.asarray(pred_b, float)
    batteries = np.asarray(batteries)
    uniq = np.array(sorted(set(batteries)))
    index = {b: np.where(batteries == b)[0] for b in uniq}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[b] for b in draw])
        estimates.append(mae(y_true[idx], pred_b[idx]) - mae(y_true[idx], pred_a[idx]))
    return float(np.mean(estimates)), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))
