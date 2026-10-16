"""Figure export: readability normalisation, typography audit, SVG/PDF/PNG.

Mirrors the reference exporter (`nc_style.export_figure`): the figure is
normalised to the Nature Communications display width, strokes and font sizes
are floored at the publication minima, an objective typography audit is written
next to the output, and all three formats are saved without rasterisation.
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.text import Text

from .paths import figure_dir
from .style import FONT_FAMILY, MM, NC_WIDTH

NC_WIDTH_MM = 179.0


def enforce_nc_readability(figure) -> None:
    """Normalise typography and strokes at the final 179-mm width.

    Deliberately conservative: data coordinates, limits, colours and artist
    visibility are untouched.
    """
    figure.canvas.draw()
    width, height = figure.get_size_inches()
    if width > 0 and abs(width - NC_WIDTH) > 1e-6:
        figure.set_size_inches(NC_WIDTH, height * NC_WIDTH / width, forward=True)

    panel_tokens = {f"({letter})" for letter in "abcdefghijklmnopqrstuvwxyz"} \
        | set("abcdefghijklmnopqrstuvwxyz")
    for axis in figure.get_axes():
        for spine in axis.spines.values():
            spine.set_linewidth(max(float(spine.get_linewidth()), 0.8))
        axis.tick_params(which="major", width=0.7)
        axis.tick_params(which="minor", width=0.55)
        axis.xaxis.label.set_fontsize(max(float(axis.xaxis.label.get_fontsize()), 6.7))
        axis.yaxis.label.set_fontsize(max(float(axis.yaxis.label.get_fontsize()), 6.7))
        axis.title.set_fontsize(max(float(axis.title.get_fontsize()), 7.2))
        if axis.title.get_text().strip():
            axis.title.set_fontweight("bold")
        for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            tick.set_fontsize(max(float(tick.get_fontsize()), 5.7))
        legend = axis.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                item.set_fontsize(max(float(item.get_fontsize()), 5.7))
        for line in axis.get_lines():
            line.set_linewidth(max(float(line.get_linewidth()), 0.8))
        for collection in axis.collections:
            collection.set_rasterized(False)
            try:
                widths = collection.get_linewidths()
                if len(widths):
                    collection.set_linewidths([max(float(v), 0.7) for v in widths])
            except (AttributeError, TypeError, ValueError):
                pass

    for artist in figure.findobj():
        if hasattr(artist, "set_rasterized"):
            try:
                artist.set_rasterized(False)
            except (AttributeError, TypeError):
                pass
        if isinstance(artist, Text):
            value = artist.get_text().strip()
            if not value:
                continue
            artist.set_fontfamily(FONT_FAMILY)
            artist.set_fontsize(max(float(artist.get_fontsize()), 5.5))
            if value in panel_tokens:
                artist.set_fontsize(8.0)
                artist.set_fontweight("bold")


def _typography_audit(figure, figure_id: str, output: Path) -> dict:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.get_window_extent(renderer)
    records, clipped = [], []
    for text in figure.findobj(match=lambda a: hasattr(a, "get_text")):
        try:
            value = text.get_text()
            visible = text.get_visible()
            size = float(text.get_fontsize())
        except Exception:
            continue
        if not visible or not str(value).strip():
            continue
        records.append({"text": str(value), "fontsize_pt": size,
                        "family": list(text.get_fontfamily())})
        try:
            box = text.get_window_extent(renderer)
            if (box.x0 < figure_box.x0 - 2.0 or box.y0 < figure_box.y0 - 2.0
                    or box.x1 > figure_box.x1 + 2.0 or box.y1 > figure_box.y1 + 2.0):
                clipped.append(str(value))
        except Exception:
            pass
    sizes = [record["fontsize_pt"] for record in records]
    payload = {
        "figure": figure_id,
        "min_font_pt": min(sizes) if sizes else None,
        "text_count": len(records),
        "below_5pt": [r for r in records if r["fontsize_pt"] < 5.0],
        "clipped_text_candidates": clipped,
        "font_fallback": FONT_FAMILY,
        "target_width_mm": NC_WIDTH_MM,
        "texts": records,
    }
    (output / f"{figure_id}_typography.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def export_figure(figure, figure_id: str, stem: str | None = None) -> dict[str, str]:
    """Normalise, audit and export one figure into `figures/<figure_id>/`."""
    stem = stem or figure_id
    output = figure_dir(figure_id)
    enforce_nc_readability(figure)
    audit = _typography_audit(figure, figure_id, output)
    paths = {suffix: output / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png")}
    figure.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    figure.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    figure.savefig(paths["png"], dpi=600, bbox_inches="tight", facecolor="white")
    (output / f"{stem}_audit.json").write_text(json.dumps({
        "figure": figure_id,
        "width_mm": float(figure.get_size_inches()[0] / MM),
        "height_mm": float(figure.get_size_inches()[1] / MM),
        "outputs": {key: str(value) for key, value in paths.items()},
        "min_font_pt": audit["min_font_pt"],
    }, indent=2) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}
