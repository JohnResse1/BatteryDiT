"""Canonical paths for the analysis layer.

Layout (enforced by `tests/regression/test_analysis_layering.py`):

    analysis/<task>.py      reads `results/`, writes `analysis/outputs/`
    figures/<figure_id>/    reads `analysis/outputs/`, writes its own folder

Statistics therefore live in `analysis/` and plotting lives in `figures/`.
"""
from __future__ import annotations

from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
RESULTS_ROOT = REPOSITORY_ROOT / "results"
OUTPUT_ROOT = ANALYSIS_ROOT / "outputs"
ANALYSIS_OUTPUT_ROOT = OUTPUT_ROOT
PAPER_RESULTS_ROOT = REPOSITORY_ROOT / "paper_results"


def create_output_directories() -> None:
    ANALYSIS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
