"""Read analysis-layer tables.  Figures never read `results/` directly."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ANALYSIS_OUTPUT_ROOT


def _path(name: str) -> Path:
    path = ANALYSIS_OUTPUT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(
            f"missing analysis table {name!r}; run the matching "
            f"`python -m analysis.<task>` first")
    return path


def read_table(name: str, **kwargs: Any) -> pd.DataFrame:
    path = _path(name)
    if path.suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path, **kwargs)


def read_json(name: str) -> Any:
    return json.loads(_path(name).read_text(encoding="utf-8"))


def read_arrays(name: str) -> dict:
    import numpy as np
    path = _path(name)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}
