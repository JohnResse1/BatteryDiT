"""Adapters for HUST, BatteryLife, SDU, and TJU datasets."""
from __future__ import annotations
from pathlib import Path
from . import batterylife, hust, sdu, tju

def build(name: str, data_root: Path, dataset_root: Path, splits: dict[str,list[str]]) -> int:
    if name == "hust": return hust.build(data_root, dataset_root / "hust", splits)
    if name in {"mich", "stanford_2"}: return batterylife.build(name, data_root, dataset_root / "batterylife", splits)
    if name == "sdu": return sdu.build(name, data_root, dataset_root / "sdu", splits)
    return tju.build(name, data_root, dataset_root / "tju", splits)
