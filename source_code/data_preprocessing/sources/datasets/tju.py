"""Build the TJU battery dataset for the BatteryDiT protocol.

Data source
-----------
Zhu, J., Wang, Y., Huang, Y., et al.,
"Data-driven capacity estimation of commercial lithium-ion batteries
from voltage relaxation,"
Nature Communications, 13, 2261 (2022).
DOI: 10.1038/s41467-022-29837-w

The TJU (Tongji University) dataset contains commercial NCA, NCM, and
NCM+NCA lithium-ion cells cycled under different temperatures and
charge/discharge rates.

The raw records are independently processed here using the BatteryDiT
resampling, corruption, and evaluation protocol.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ...corruptions import apply
from ...resampling import resample_charge
from ...schema import sample_path, write_sample

DATASET_SOURCES = {
    "tju_nca_45c": ("Dataset_1_NCA_battery", "NCA", 3.5),
    "tju_ncm_45c": ("Dataset_2_NCM_battery", "NCM", 3.5),
    "tju_ncm_nca_25c": ("Dataset_3_NCM_NCA_battery", "NCA_NCM", 2.5),
}
TEST_CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")


def _seed(dataset: str, split: str, cell: str, cycle: int, condition: str) -> int:
    key = f"{dataset}/{split}/{cell}/{cycle}/{condition}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def _read_cycles(path: Path) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
    """Extract valid charging curves and their charge capacity from one raw CSV."""
    frame = pd.read_csv(path)
    columns = {name.lower().replace(" ", ""): name for name in frame.columns}
    cycle_column = columns["cyclenumber"]
    current_column = columns["<i>/ma"]
    voltage_column = columns["ecell/v"]
    capacity_column = columns["qcharge/ma.h"]

    records = []
    for raw_cycle, cycle_frame in frame.groupby(cycle_column, sort=True):
        cycle = int(raw_cycle)
        current = cycle_frame[current_column].to_numpy(dtype=np.float32) / 1000.0
        voltage = cycle_frame[voltage_column].to_numpy(dtype=np.float32)
        charge_capacity = cycle_frame[capacity_column].to_numpy(dtype=np.float32)
        charging = np.isfinite(current) & np.isfinite(voltage) & (current > 1e-4)
        if charging.sum() < 2:
            continue
        capacity = float(np.nanmax(charge_capacity)) / 1000.0
        if not np.isfinite(capacity):
            continue
        try:
            current, voltage = resample_charge(current[charging], voltage[charging])
        except ValueError:
            continue
        records.append((cycle, current, voltage, capacity))
    return records


def build(dataset: str, data_root: Path, source_root: Path, splits: dict[str, list[str]]) -> int:
    """Build selected TJU cells with the shared formal schema and corruptions."""
    try:
        csv_directory, prefix, nominal_capacity = DATASET_SOURCES[dataset]
    except KeyError as error:
        raise ValueError(f"unsupported TJU dataset: {dataset}") from error

    paths = {
        f"{prefix}_{path.stem}": path
        for path in (source_root / csv_directory).glob("*.csv")
    }
    split_for_cell = {cell: split for split, cells in splits.items() for cell in cells}
    missing = sorted(set(split_for_cell) - set(paths))
    if missing:
        raise KeyError(f"TJU raw CSV missing requested cells: {missing}")

    written = 0
    for cell, split in sorted(split_for_cell.items()):
        records = _read_cycles(paths[cell])
        if len(records) < 2:
            raise ValueError(f"TJU {cell} has fewer than two valid charging cycles")
        first_cycle, last_cycle = records[0][0], records[-1][0]
        life_span = last_cycle - first_cycle
        if life_span <= 0:
            raise ValueError(f"TJU {cell} has invalid cycle range")
        conditions = TEST_CONDITIONS if split in {"vali", "test"} else ("clean",)
        for cycle, current, voltage, capacity in records:
            for condition in conditions:
                current_out, voltage_out = apply(condition, current, voltage, _seed(dataset, split, cell, cycle, condition))
                write_sample(
                    sample_path(data_root, dataset, split, condition, cell, cycle),
                    current_out,
                    voltage_out,
                    capacity / nominal_capacity,
                    (last_cycle - cycle) / life_span,
                )
                written += 1
    return written
