"""Read formal experiment outputs without relying on historical path names."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .paths import RESULTS_ROOT


FORMAL_CONDITIONS = ("clean", "downsample", "segment", "random", "destructive")
FORMAL_SEEDS = (2021, 2022, 2023)


@dataclass(frozen=True)
class FormalRun:
    """One completed experiment run and its machine-readable outputs."""

    group: str
    dataset: str
    setting: str
    directory: Path
    config: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def arguments(self) -> dict[str, Any]:
        return self.config.get("arguments", {})

    @property
    def seed(self) -> int:
        return int(self.arguments.get("seed", -1))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON output: {path}") from error


def completed_runs(group: str, dataset: str, *, results_root: Path = RESULTS_ROOT) -> list[FormalRun]:
    """Return completed runs in deterministic setting order."""
    root = Path(results_root) / group / dataset
    if not root.is_dir():
        return []
    runs: list[FormalRun] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = directory / "config.json"
        metrics_path = directory / "test_metrics.json"
        if not metrics_path.is_file():
            continue
        config = _read_json(config_path) if config_path.is_file() else {"arguments": {}}
        runs.append(FormalRun(group, dataset, directory.name, directory, config, _read_json(metrics_path)))
    return runs


def select_runs(
    group: str,
    dataset: str,
    *,
    recon_model: str | None = None,
    direct_model: str | None = None,
    predictor: str | None = None,
    task_name: str | None = None,
    seeds: Iterable[int] | None = FORMAL_SEEDS,
    results_root: Path = RESULTS_ROOT,
) -> list[FormalRun]:
    """Select completed runs by the arguments recorded during training."""
    selected = []
    seed_set = set(seeds) if seeds is not None else None
    for run in completed_runs(group, dataset, results_root=results_root):
        arguments = run.arguments
        if seed_set is not None and run.seed not in seed_set:
            continue
        if recon_model is not None and arguments.get("recon_model") != recon_model:
            continue
        if direct_model is not None and arguments.get("direct_model") != direct_model:
            continue
        if predictor is not None and arguments.get("predictor", "trb") != predictor:
            continue
        if task_name is not None and arguments.get("task_name") != task_name:
            continue
        selected.append(run)
    return sorted(selected, key=lambda run: run.seed)


def require_runs(runs: list[FormalRun], description: str, expected: int) -> list[FormalRun]:
    found = [run.seed for run in runs]
    expected_seeds = list(FORMAL_SEEDS[:expected])
    if len(runs) != expected or found != expected_seeds:
        raise FileNotFoundError(
            f"{description} requires completed seeds {expected_seeds}; found {found}. "
            "Wait for the matching SLURM tasks and check test_metrics.json."
        )
    return runs


def condition_matrix(runs: list[FormalRun], metric: str) -> np.ndarray:
    """Return a seed-by-condition metric matrix in the formal condition order."""
    rows = []
    for run in runs:
        conditions = run.metrics.get("conditions", {})
        missing = set(FORMAL_CONDITIONS) - set(conditions)
        if missing:
            raise ValueError(f"{run.directory} is missing conditions {sorted(missing)}")
        try:
            rows.append([float(conditions[name][metric]) for name in FORMAL_CONDITIONS])
        except KeyError as error:
            raise ValueError(f"{run.directory} does not contain metric {metric!r}") from error
    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {metric} values found in {[str(run.directory) for run in runs]}")
    return values


def mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"expected a non-empty seed-by-condition matrix, got {values.shape}")
    deviation = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1])
    return values.mean(axis=0), deviation
