"""Load the formal per-cycle BatteryDiT data schema."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
FORMAL_DATASETS = (
    "mit",
    "hust",
    "mich",
    "sdu",
    "stanford_2",
    "tju_nca_45c",
    "tju_ncm_45c",
    "tju_ncm_nca_25c",
)
LEGACY_ALIASES = {"dataset_MIT": "mit", "dataset_HUST": "hust"}
SPLITS = ("train", "vali", "test")
CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")
SAMPLE_FIELDS = ("I", "V", "SOH", "RUL")
SEQUENCE_LENGTH = 1000


def dataset_name(value: str) -> str:
    """Resolve a dataset name to one unambiguous formal dataset."""
    name = LEGACY_ALIASES.get(value, value)
    if name not in FORMAL_DATASETS:
        choices = ", ".join(FORMAL_DATASETS)
        raise ValueError(f"unknown formal dataset {value!r}; choose one of: {choices}")
    return name


def condition_name(dataset: str, split: str, value: str | None) -> str:
    """Resolve a historical condition spelling to the formal directory name."""
    if split == "train":
        return "clean"
    condition = (value or "clean").lower()
    for prefix in ("vali_", "test_", "tju_test_"):
        if condition.startswith(prefix):
            condition = condition[len(prefix):]
    if condition == "origin":
        condition = "clean"
    if condition not in CONDITIONS:
        choices = ", ".join(CONDITIONS)
        raise ValueError(f"unknown condition {value!r}; choose one of: {choices}")
    return condition


def _sample_paths(root: Path) -> list[Path]:
    paths = list(root.rglob("*.npz"))
    paths.sort(key=lambda path: (path.parent.name, int(path.stem)))
    return paths


def _tensor(sample: np.lib.npyio.NpzFile, field: str) -> torch.Tensor:
    value = np.asarray(sample[field], dtype=np.float32)
    expected = (SEQUENCE_LENGTH, 1) if field in {"I", "V"} else (1,)
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError(f"invalid {field} in {sample.fid.name}: expected {expected}, got {value.shape}")
    return torch.from_numpy(value)


class BatteryDataset(Dataset):
    """One formal dataset, split, and corruption condition.

    Each item is returned as ``(I, V, SOH, RUL)`` with the final data schema.
    A loader never combines datasets with different chemistry or label policies.
    """

    def __init__(
        self,
        dataset: str,
        split: str = "train",
        condition: str | None = None,
        data_root: Path | str = DATA_ROOT,
    ):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; choose one of: {', '.join(SPLITS)}")
        self.dataset = dataset_name(dataset)
        self.split = split
        self.condition = condition_name(self.dataset, split, condition)
        self.root = Path(data_root) / self.dataset / split / self.condition
        if not self.root.is_dir():
            split_root = Path(data_root) / self.dataset / split
            available = sorted(path.name for path in split_root.iterdir() if path.is_dir()) if split_root.is_dir() else []
            raise FileNotFoundError(
                f"{self.dataset}/{split}/{self.condition} has not been built; "
                f"available conditions: {', '.join(available) or 'none'}"
            )
        self.paths = _sample_paths(self.root)
        if not self.paths:
            raise FileNotFoundError(f"no NPZ samples under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        with np.load(path) as sample:
            if set(sample.files) != set(SAMPLE_FIELDS):
                raise ValueError(f"{path} has fields {sample.files}; expected {SAMPLE_FIELDS}")
            return tuple(_tensor(sample, field) for field in SAMPLE_FIELDS)  # type: ignore[return-value]


class PairedBatteryDataset(Dataset):
    """Corrupted samples paired with their clean counterpart by relative path."""

    def __init__(
        self,
        dataset: str,
        split: str,
        condition: str,
        data_root: Path | str = DATA_ROOT,
    ):
        self.inputs = BatteryDataset(dataset, split, condition, data_root)
        self.targets = BatteryDataset(dataset, split, "clean", data_root)
        input_keys = {path.relative_to(self.inputs.root): path for path in self.inputs.paths}
        target_keys = {path.relative_to(self.targets.root): path for path in self.targets.paths}
        if input_keys.keys() != target_keys.keys():
            missing_target = sorted(str(key) for key in input_keys.keys() - target_keys.keys())
            missing_input = sorted(str(key) for key in target_keys.keys() - input_keys.keys())
            raise ValueError(
                f"{self.inputs.dataset}/{split}/{self.inputs.condition} is not paired with clean; "
                f"missing clean={missing_target[:3]}, missing corrupt={missing_input[:3]}"
            )
        self.keys = sorted(input_keys)
        self.input_paths = [input_keys[key] for key in self.keys]
        self.target_paths = [target_keys[key] for key in self.keys]
        self.dataset = self.inputs.dataset
        self.split = split
        self.condition = self.inputs.condition

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        with np.load(self.input_paths[index]) as input_sample, np.load(self.target_paths[index]) as target_sample:
            current = _tensor(input_sample, "I")
            voltage = _tensor(input_sample, "V")
            clean_voltage = _tensor(target_sample, "V")
            soh = _tensor(target_sample, "SOH")
            rul = _tensor(target_sample, "RUL")
        return current, voltage, clean_voltage, soh, rul, str(self.keys[index])
