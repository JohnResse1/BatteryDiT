"""Paths used by the figure layer.

Figures live outside `analysis/`:

    figures/<figure_id>/plot.py      drawing code
    figures/<figure_id>/<stem>.png   that figure's own output

Shared plotting rules are in `figures/common/`; the analysis products they read
are written by the analysis layer into `analysis/outputs/`.
"""
from __future__ import annotations

from pathlib import Path

FIGURE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FIGURE_ROOT.parent
ANALYSIS_OUTPUT_ROOT = REPOSITORY_ROOT / "analysis" / "outputs"
PAPER_RESULTS_ROOT = REPOSITORY_ROOT / "paper_results"


def figure_dir(figure_id: str) -> Path:
    """Output directory for one figure, created on demand."""
    path = FIGURE_ROOT / figure_id
    path.mkdir(parents=True, exist_ok=True)
    return path
