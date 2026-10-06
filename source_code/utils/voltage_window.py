"""Small formal voltage-window feature used by degradation signatures."""
from __future__ import annotations

import numpy as np


def _to_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {values.shape}")
    return values


def _interpolate_threshold_index(values: np.ndarray, threshold: float) -> np.ndarray:
    """Return the fractional index at which each curve reaches ``threshold``."""
    _, length = values.shape
    result = np.empty(len(values), dtype=np.float64)
    for index, curve in enumerate(values):
        reached = curve >= threshold
        first = np.argmax(reached)
        if not reached[first]:
            result[index] = float(length - 1)
        elif first == 0:
            result[index] = 0.0
        else:
            lower, upper = curve[first - 1], curve[first]
            if upper - lower < 1e-12:
                result[index] = float(first)
            else:
                fraction = (threshold - lower) / (upper - lower)
                result[index] = float(first - 1) + fraction
    return result


def delta_Q(
    values: np.ndarray,
    V_low: float | None = None,
    V_high: float | None = None,
) -> np.ndarray:
    """Return the normalized area between two voltage thresholds."""
    values = _to_2d(values)
    batch, length = values.shape
    total_area = np.trapz(values, axis=1)
    if V_low is None:
        V_low = np.percentile(values, 10)
    if V_high is None:
        V_high = np.percentile(values, 90)

    lower = np.clip(
        _interpolate_threshold_index(values, V_low), 0, length - 1
    )
    upper = np.clip(
        _interpolate_threshold_index(values, V_high), 0, length - 1
    )
    window_area = np.zeros(batch, dtype=np.float64)
    for index in range(batch):
        start = max(0, min(int(np.floor(lower[index])), length - 2))
        stop = max(start + 1, min(int(np.ceil(upper[index])), length - 1))
        window_area[index] = np.trapz(values[index, start : stop + 1])

    ratio = np.where(total_area > 1e-12, window_area / total_area, 0.0)
    return ratio[:, np.newaxis]
