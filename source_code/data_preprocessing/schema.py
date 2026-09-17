"""Validation and atomic writing for final per-cycle NPZ samples."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


SEQUENCE_LENGTH = 1000
SAMPLE_FIELDS = ("I", "V", "SOH", "RUL")


def curve(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (SEQUENCE_LENGTH,):
        array = array[:, None]
    if array.shape != (SEQUENCE_LENGTH, 1):
        raise ValueError(f"{name} must have shape ({SEQUENCE_LENGTH}, 1), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def scalar(value: np.ndarray | float, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (1,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite scalar, got {array}")
    return array


def sample_path(data_root: Path, dataset: str, split: str, condition: str, cell: str, cycle: int) -> Path:
    if split not in {"train", "vali", "test"}:
        raise ValueError(f"unknown split: {split!r}")
    return data_root / dataset / split / condition / cell / f"{int(cycle)}.npz"


def write_sample(
    path: Path,
    current: np.ndarray,
    voltage: np.ndarray,
    soh: np.ndarray | float,
    rul: np.ndarray | float,
) -> None:
    """Write one complete sample atomically; no other module writes NPZ files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, I=curve(current, "I"), V=curve(voltage, "V"), SOH=scalar(soh, "SOH"), RUL=scalar(rul, "RUL"))
    os.replace(temporary, path)


def validate_sample(path: Path) -> None:
    with np.load(path) as archive:
        if set(archive.files) != set(SAMPLE_FIELDS):
            raise ValueError(f"{path} has fields {archive.files}, expected {SAMPLE_FIELDS}")
        curve(archive["I"], "I")
        curve(archive["V"], "V")
        scalar(archive["SOH"], "SOH")
        scalar(archive["RUL"], "RUL")
