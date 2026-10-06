"""Deterministic table IO shared by the analysis and figure layers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ANALYSIS_OUTPUT_ROOT, create_output_directories


def analysis_dir() -> Path:
    create_output_directories()
    return ANALYSIS_OUTPUT_ROOT


def write_table(frame: pd.DataFrame, name: str, **kwargs: Any) -> Path:
    """Write one analysis table into `outputs/analysis/` (CSV)."""
    path = analysis_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".csv"
    if suffix == ".csv":
        frame.to_csv(path, index=kwargs.pop("index", False), **kwargs)
    elif suffix in {".json", ".gz"}:
        frame.to_csv(path, index=kwargs.pop("index", False), **kwargs)
    else:
        raise ValueError(f"unsupported table suffix {suffix!r} for {name!r}")
    return path


def write_json(payload: Any, name: str) -> Path:
    path = analysis_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_table(name: str, **kwargs: Any) -> pd.DataFrame:
    """Read one analysis table written by the analysis layer."""
    path = ANALYSIS_OUTPUT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(
            f"missing analysis table {name!r}; run the analysis layer first "
            f"(python main.py --analysis <task>)")
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, **kwargs)
    if path.suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"unsupported table suffix for {name!r}")


def read_json(name: str) -> Any:
    path = ANALYSIS_OUTPUT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(
            f"missing analysis record {name!r}; run the analysis layer first")
    return json.loads(path.read_text(encoding="utf-8"))


def has_table(name: str) -> bool:
    return (ANALYSIS_OUTPUT_ROOT / name).is_file()
