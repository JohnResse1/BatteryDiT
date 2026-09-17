"""Canonical paths used by every paper-analysis task."""
from __future__ import annotations

from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
RESULTS_ROOT = REPOSITORY_ROOT / "results"
OUTPUT_ROOT = ANALYSIS_ROOT / "outputs"
MAIN_FIGURE_ROOT = OUTPUT_ROOT / "main"
SUPPLEMENTARY_FIGURE_ROOT = OUTPUT_ROOT / "supplementary"
AUDIT_ROOT = OUTPUT_ROOT / "audit"


def create_output_directories() -> None:
    for path in (MAIN_FIGURE_ROOT, SUPPLEMENTARY_FIGURE_ROOT, AUDIT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
