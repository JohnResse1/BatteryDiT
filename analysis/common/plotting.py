"""Shared Nature Communications figure style and export helpers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from .paths import AUDIT_ROOT, MAIN_FIGURE_ROOT, SUPPLEMENTARY_FIGURE_ROOT, create_output_directories


MM = 1.0 / 25.4
FIGURE_WIDTH = 179.0 * MM
COLORS = {
    "batterydit": "#2166AC",
    "csdi": "#9BAFBD",
    "transformer": "#766C9A",
    "kalman": "#D6604D",
    "tcn": "#E59A52",
    "lstm": "#4C9F70",
    "gru": "#8E6BBE",
    "trb": "#333333",
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 6.0,
        "axes.labelsize": 6.7,
        "axes.titlesize": 7.2,
        "axes.titleweight": "bold",
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.1,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(axis, label: str, x: float = -0.10, y: float = 1.05) -> None:
    axis.text(x, y, f"({label})", transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")


def export_figure(figure, stem: str, *, supplementary: bool = False) -> dict[str, str]:
    """Export an editable SVG/PDF and a 600-dpi PNG from the same figure."""
    create_output_directories()
    output = SUPPLEMENTARY_FIGURE_ROOT if supplementary else MAIN_FIGURE_ROOT
    paths = {suffix: output / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png")}
    for suffix, path in paths.items():
        figure.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    audit = {
        "figure": stem,
        "width_mm": float(figure.get_size_inches()[0] / MM),
        "height_mm": float(figure.get_size_inches()[1] / MM),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    (AUDIT_ROOT / f"{stem}.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}
