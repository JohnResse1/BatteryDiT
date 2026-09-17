"""Shared charge-curve orientation and 1000-point interpolation."""
from __future__ import annotations

import numpy as np

from .schema import SEQUENCE_LENGTH


def resample_charge(current: np.ndarray, voltage: np.ndarray, coordinate: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Orient a charge curve and linearly resample it once to 1000 points."""
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    voltage = np.asarray(voltage, dtype=np.float64).reshape(-1)
    if coordinate is None:
        coordinate = np.arange(len(current), dtype=np.float64)
    else:
        coordinate = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    valid = np.isfinite(current) & np.isfinite(voltage) & np.isfinite(coordinate)
    current, voltage, coordinate = current[valid], voltage[valid], coordinate[valid]
    if len(current) < 20:
        raise ValueError("a charge curve needs at least 20 valid points")
    if float(np.nanmedian(current)) < 0:
        current = -current
    order = np.argsort(coordinate, kind="stable")
    coordinate, current, voltage = coordinate[order], current[order], voltage[order]
    coordinate, unique = np.unique(coordinate, return_index=True)
    current, voltage = current[unique], voltage[unique]
    if len(coordinate) < 20 or coordinate[-1] <= coordinate[0]:
        raise ValueError("charge coordinate must contain increasing values")
    target = np.linspace(coordinate[0], coordinate[-1], SEQUENCE_LENGTH)
    return (
        np.interp(target, coordinate, current).astype(np.float32)[:, None],
        np.interp(target, coordinate, voltage).astype(np.float32)[:, None],
    )
