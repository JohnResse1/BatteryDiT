"""Build the HUST battery dataset for the BatteryDiT protocol.

Data source
-----------
Ma, G., Xu, S., Jiang, B., et al.,
"Real-time personalized health status prediction of lithium-ion batteries
using deep transfer learning,"
Energy & Environmental Science, 15, 4083–4094 (2022).
DOI: 10.1039/D2EE01676A

The original dataset contains 77 commercial LFP/graphite cells operated
under different multi-stage discharge protocols.

The raw records are independently processed here using the BatteryDiT
resampling, corruption, and evaluation protocol.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from ...corruptions import apply
from ...datasets import protocol
from ...resampling import resample_charge
from ...schema import sample_path, write_sample


EVALUATION_CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")


def _seed(split: str, cell: str, cycle: int, condition: str) -> int:
    value = f"hust/{split}/{cell}/{cycle}/{condition}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little")


def _load(source_root: Path, cell: str) -> dict:
    with (source_root / f"{cell}.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    return payload[cell]


def _charge_curve(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"Status", "Current (mA)", "Voltage (V)", "Time (s)"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"HUST cycle is missing columns: {sorted(missing)}")
    # HUST labels its discharge stages as ``Constant current discharge_*``.
    # A substring search for "charge" therefore also selects every discharge
    # row.  Match the two documented charging stages explicitly.
    status = frame["Status"].astype(str).str.strip().str.lower()
    mask = status.isin({
        "constant current charge",
        "constant current-constant voltage charge",
    })
    charge = frame.loc[mask, ["Current (mA)", "Voltage (V)", "Time (s)"]].dropna().sort_values("Time (s)")
    return (
        charge["Current (mA)"].to_numpy(float) / 1000.0,
        charge["Voltage (V)"].to_numpy(float),
        charge["Time (s)"].to_numpy(float),
    )


def _write_cell(data_root: Path, split: str, cell: str, source: dict) -> int:
    cycles = source["data"]
    valid_cycles: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cycle, frame in cycles.items():
        current, voltage, time = _charge_curve(frame)
        if len(current) < 20:
            continue
        try:
            clean_i, clean_v = resample_charge(current, voltage, time)
        except ValueError:
            continue
        valid_cycles.append((int(cycle), clean_i, clean_v))
    valid_cycles.sort(key=lambda value: value[0])
    if len(valid_cycles) < 2:
        raise ValueError(f"HUST {cell} has fewer than two usable charge cycles")
    first_cycle, last_cycle = valid_cycles[0][0], valid_cycles[-1][0]
    life_span = last_cycle - first_cycle
    written = 0
    for cycle, clean_i, clean_v in valid_cycles:
        soh = protocol("hust").soh(float(source["dq"][cycle]) / 1000.0)
        rul = (last_cycle - cycle) / life_span
        conditions = EVALUATION_CONDITIONS if split in {"vali", "test"} else ("clean",)
        for condition in conditions:
            current_out, voltage_out = apply(condition, clean_i, clean_v, _seed(split, cell, int(cycle), condition))
            write_sample(sample_path(data_root, "hust", split, condition, cell, int(cycle)), current_out, voltage_out, soh, rul)
            written += 1
    return written


def build(data_root: Path, source_root: Path, splits: dict[str, list[str]]) -> int:
    """Build HUST with frozen splits and the shared post-resampling corruptions."""
    written = 0
    for split, cells in splits.items():
        for cell in cells:
            written += _write_cell(data_root, split, cell, _load(source_root, cell))
    return written
