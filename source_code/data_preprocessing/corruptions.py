"""Formal evaluation corruption which may occur in real-world scenarios.
input: 1000-point curves (current, voltage)
output: 1000-point curves (current, voltage) with one of the following conditions:
- clean: no corruption
- downsample: only 600 points are observed, the rest are interpolated
- segment: a contiguous 600-point segment is missing, the rest are observed
- random: each point is observed with 75% probability, the rest are missing
- destructive: voltage is corrupted with noise, drift, and spikes; current is corrupted with noise

"""
from __future__ import annotations

import numpy as np


CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")


def apply(condition: str, current: np.ndarray, voltage: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply one deterministic corruption to already-resampled 1000-point curves."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition!r}")
    current = np.asarray(current, dtype=np.float32).copy()
    voltage = np.asarray(voltage, dtype=np.float32).copy()
    if current.shape != voltage.shape or current.shape != (1000, 1):
        raise ValueError(f"expected matching (1000, 1) curves, got {current.shape} and {voltage.shape}")
    if condition == "clean":
        return current, voltage
    rng = np.random.default_rng(seed)
    if condition == "downsample":
        kept = np.unique(np.rint(np.linspace(0, 999, 600)).astype(int))
        target = np.arange(1000)
        current[:, 0] = np.interp(target, kept, current[kept, 0])
        voltage[:, 0] = np.interp(target, kept, voltage[kept, 0])
    elif condition == "segment":
        start = int(rng.integers(0, 401))
        current[start : start + 600] = 0.0
        voltage[start : start + 600] = 0.0
    elif condition == "random":
        observed = (rng.random((1000, 1)) >= 0.75).astype(np.float32)
        current *= observed
        voltage *= observed
    else:
        voltage += rng.normal(0.0, 0.1, voltage.shape).astype(np.float32)
        voltage += np.cumsum(rng.normal(0.0, 0.005, voltage.shape).astype(np.float32), axis=0)
        spike = rng.random(voltage.shape) < 0.005
        direction = np.where(rng.normal(size=voltage.shape) >= 0.0, 1.0, -1.0)
        magnitude = rng.uniform(1.0, 2.5, voltage.shape)
        voltage += (spike * direction * magnitude).astype(np.float32)
        current += rng.normal(0.0, float(np.abs(current).max()) * 0.005 + 5e-9, current.shape).astype(np.float32)
    return current, voltage
