"""Shared canonical workflow provenance and output contracts.

Each formal workflow writes into:
    outputs/canonical/<workflow>/<run_id>/{raw,analysis,plotting,provenance}

The plotting directory contains only plotting-ready tables. Analysis code
never writes there. A run manifest captures git commit, configuration, seeds,
input result IDs and schema version.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.utils.paths import ANALYSIS_OUTPUT_ROOT

SCHEMA_VERSION = "canonical-r3-r5-v1"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class CanonicalRun:
    workflow: str
    run_id: str
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def plotting(self) -> Path:
        return self.root / "plotting"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance"

    def write_table(self, layer: str, name: str, frame: pd.DataFrame) -> Path:
        if layer not in {"raw", "analysis", "plotting", "provenance"}:
            raise ValueError("layer must be raw, analysis, plotting, or provenance")
        path = getattr(self, layer) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz" or path.name.endswith(".csv.gz"):
            frame.to_csv(path, index=False, compression="gzip")
        elif path.suffix == ".csv":
            frame.to_csv(path, index=False)
        else:
            raise ValueError("write_table supports only CSV/CSV.GZ")
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.provenance / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        return path

    def write_manifest(
        self,
        *,
        config: dict[str, Any],
        input_paths: list[str] | dict[str, str],
        checkpoint_ids: dict[str, str] | None = None,
        seeds: list[int] | None = None,
        split_manifest: str | None = None,
    ) -> Path:
        payload = {
            "run_id": self.run_id,
            "workflow": self.workflow,
            "schema_version": SCHEMA_VERSION,
            "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "input_result_ids": input_paths,
            "checkpoint_ids": checkpoint_ids or {},
            "random_seed": seeds,
            "split_manifest": split_manifest,
        }
        return self.write_json("run_manifest.json", payload)


def canonical_run(workflow: str, run_id: str | None = None) -> CanonicalRun:
    """Create an isolated canonical output run."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ANALYSIS_OUTPUT_ROOT / "canonical" / workflow / run_id
    for subdirectory in ("raw", "analysis", "plotting", "provenance"):
        (root / subdirectory).mkdir(parents=True, exist_ok=True)
    return CanonicalRun(workflow, run_id, root)
