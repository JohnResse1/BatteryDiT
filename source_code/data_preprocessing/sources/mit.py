"""Build the MIT battery dataset for the BatteryDiT protocol.

Data source
-----------
Severson, K. A., Attia, P. M., Jin, N., et al.,
"Data-driven prediction of battery cycle life before capacity degradation,"
Nature Energy, 4, 383–391 (2019).
DOI: 10.1038/s41560-019-0356-8

The original dataset contains commercial LFP/graphite cells cycled under
different fast-charging protocols.

The raw records are independently processed here using the BatteryDiT
resampling, corruption, and evaluation protocol.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..corruptions import apply
from ..datasets import protocol
from ..resampling import resample_charge
from ..schema import sample_path, write_sample


EVALUATION_CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")


def _seed(split: str, cell: str, cycle: int, condition: str) -> int:
    value = f"mit/{split}/{cell}/{cycle}/{condition}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little")


def _load(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"MIT source is not a cell dictionary: {path}")
    return data


def _write_cell(
    data_root: Path,
    split: str,
    cell: str,
    source: dict[str, Any],
) -> int:
    cycles = source["cycles"]
    summary = source["summary"]
    qd = np.asarray(summary["QD"], dtype=float)
    valid_cycles: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cycle_text, record in cycles.items():
        cycle = int(cycle_text)
        if cycle == 0:
            continue
        current = np.asarray(record["I"], dtype=float)
        voltage = np.asarray(record["V"], dtype=float)
        charging = current > 1e-4
        try:
            clean_i, clean_v = resample_charge(current[charging], voltage[charging])
        except ValueError:
            continue
        valid_cycles.append((cycle, clean_i, clean_v))
    valid_cycles.sort(key=lambda value: value[0])
    if len(valid_cycles) < 2:
        raise ValueError(f"MIT {cell} has fewer than two usable charge cycles")
    first_cycle, last_cycle = valid_cycles[0][0], valid_cycles[-1][0]
    life_span = last_cycle - first_cycle
    written = 0

    for cycle, clean_i, clean_v in valid_cycles:
        if cycle >= len(qd):
            raise ValueError(f"MIT QD missing at {cell}/{cycle}")
        soh = protocol("mit").soh(float(qd[cycle]))
        rul = (last_cycle - cycle) / life_span
        conditions = EVALUATION_CONDITIONS if split in {"vali", "test"} else ("clean",)
        for condition in conditions:
            current_out, voltage_out = apply(condition, clean_i, clean_v, _seed(split, cell, cycle, condition))
            write_sample(sample_path(data_root, "mit", split, condition, cell, cycle), current_out, voltage_out, soh, rul)
            written += 1
    return written


def build(data_root: Path, source_root: Path, splits: dict[str, list[str]]) -> int:
    """Build MIT using fixed cell splits and the final corruption protocol."""
    train = _load(source_root / "train_data.pkl")
    held_out = _load(source_root / "vali_data.pkl")
    sources = {"train": train, "vali": held_out, "test": held_out}
    written = 0
    for split, cells in splits.items():
        for cell in cells:
            try:
                source = sources[split][cell]
            except KeyError as error:
                raise KeyError(f"MIT {split} cell {cell!r} is absent from its raw source") from error
            written += _write_cell(data_root, split, cell, source)
    return written
