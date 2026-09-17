"""Build MICH and Stanford_2 directly from BatteryLife pickle archives.

The source archives are distributed with BatteryLife:
Tan et al., "BatteryLife: A Comprehensive Dataset and Benchmark for Battery
Life Prediction," KDD 2025. DOI: 10.1145/3711896.3737372.

The archived records are independently processed here using the BatteryDiT
resampling, corruption, and evaluation protocol.
"""
from __future__ import annotations

import hashlib
import pickle
import zipfile
from pathlib import Path

import numpy as np

from ...corruptions import apply
from ...datasets import protocol
from ...resampling import resample_charge
from ...schema import sample_path, write_sample


ARCHIVES = {"mich": "MICH.zip", "stanford_2": "Stanford_2.zip"}
CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")


def _seed(dataset: str, split: str, cell: str, cycle: int, condition: str) -> int:
    key = f"{dataset}/{split}/{cell}/{cycle}/{condition}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def _records(raw: dict) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
    nominal = float(raw["nominal_capacity_in_Ah"])
    result = []
    seen = set()
    for fallback, cycle_data in enumerate(raw["cycle_data"], start=1):
        cycle = int(cycle_data.get("cycle_number", fallback))
        if cycle in seen:
            continue
        current = np.asarray(cycle_data.get("current_in_A", []), dtype=float)
        voltage = np.asarray(cycle_data.get("voltage_in_V", []), dtype=float)
        time = np.asarray(cycle_data.get("time_in_s", []), dtype=float)
        capacity_values = np.asarray(cycle_data.get("discharge_capacity_in_Ah", []), dtype=float)
        if current.size != voltage.size or time.size != current.size:
            continue
        capacity_values = capacity_values[np.isfinite(capacity_values)]
        if not capacity_values.size:
            continue
        capacity = float(capacity_values.max())
        if not 0.20 * nominal <= capacity <= 1.30 * nominal:
            continue
        charge = current >= max(1e-5, 0.01 * nominal)
        try:
            current_out, voltage_out = resample_charge(current[charge], voltage[charge], time[charge])
        except ValueError:
            continue
        result.append((cycle, current_out, voltage_out, capacity))
        seen.add(cycle)
    return sorted(result, key=lambda item: item[0])


def build(dataset: str, data_root: Path, archive_root: Path, splits: dict[str, list[str]]) -> int:
    """Write one formal BatteryLife dataset from its original archive."""
    archive = archive_root / ARCHIVES[dataset]
    if not archive.is_file():
        raise FileNotFoundError(archive)
    requested = {cell for cells in splits.values() for cell in cells}
    split_for_cell = {cell: split for split, cells in splits.items() for cell in cells}
    count = 0
    with zipfile.ZipFile(archive) as source:
        members = {Path(name).stem: name for name in source.namelist() if name.endswith(".pkl")}
        missing = requested - set(members)
        if missing:
            raise KeyError(f"{dataset} cells absent from archive: {sorted(missing)[:10]}")
        for cell in sorted(requested):
            raw = pickle.loads(source.read(members[cell]))
            records = _records(raw)
            if len(records) < 2:
                raise ValueError(f"{dataset}/{cell} has fewer than two usable cycles")
            first, last = records[0][0], records[-1][0]
            split = split_for_cell[cell]
            conditions = CONDITIONS if split in {"vali", "test"} else ("clean",)
            for cycle, current, voltage, capacity in records:
                soh = protocol(dataset).soh(capacity, float(raw["nominal_capacity_in_Ah"]))
                rul = (last - cycle) / (last - first)
                for condition in conditions:
                    current_out, voltage_out = apply(condition, current, voltage, _seed(dataset, split, cell, cycle, condition))
                    write_sample(sample_path(data_root, dataset, split, condition, cell, cycle), current_out, voltage_out, soh, rul)
                    count += 1
    return count
