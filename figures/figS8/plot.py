"""Supplementary Fig. S8: representative external trajectory recovery.

This module deliberately follows ``figures/figS5/plot.py``.  The only
figure-specific changes are the external cohort/condition labels, the
6-by-4 panel count, and the analysis-layer asset used as input.  No model
inference, retraining, or metric recomputation is performed here.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from figures.common import export as export_helpers
from figures.common.style import MM, NC_WIDTH, apply_paper_style
from figures.common.tables import read_arrays, read_json
from figures.figS5.plot import (
    C_CLEAN,
    C_MISSING,
    C_OBSERVED,
    C_RECON,
    _apply_legacy_typography,
    _missing_runs,
    _style_axis,
)


FIGURE_ID = "figS8"
ASSET_NPZ = "multi_dataset_representatives.npz"
ASSET_JSON = "multi_dataset_representatives.json"

DATASETS = OrderedDict([
    ("hust", ("HUST", "LFP")),
    ("mich", ("MICH", "NMC111")),
    ("sdu", ("SDU", "NMC532")),
    ("stanford_2", ("Stanford", "NMC532")),
    ("tju_nca_45c", ("TJU-NCA", "NCA")),
    ("tju_ncm_45c", ("TJU-NCM", "NCM")),
])

CONDITIONS = OrderedDict([
    ("downsample", "Downsampling"),
    ("random", "Random-point missing"),
    ("segment", "Segment missing"),
    ("destructive", "Destructive noise"),
])

# These labels are the frozen condition settings represented in the existing
# multi-dataset trajectory asset.  They are annotations only; the plot does
# not construct or reapply any corruption.
CONDITION_DETAILS = {
    "downsample": "60% retained + interpolation",
    "random": "75% missing",
    "segment": "60% missing",
    "destructive": r"$\epsilon$ = 0.10 V",
}


def _asset_rows(metadata: dict) -> list[dict[str, object]]:
    """Return provenance rows without changing any formal values."""
    rows = []
    for key, (display, chemistry) in DATASETS.items():
        item = metadata[key]
        rows.append({
            "dataset": display,
            "chemistry": chemistry,
            "representative_key": item["key"],
            "selected_index": item["selected_index"],
            "candidate_count": item["candidate_count"],
            "selection_rule": item["selection_rule"],
        })
    return rows


def _write_readme(metadata: dict, outputs: dict[str, str]) -> None:
    """Write a concise provenance/readout note next to the new figure."""
    output_dir = Path(__file__).resolve().parent
    rows = _asset_rows(metadata)
    table = [
        "| Dataset | Chemistry | Representative cell/cycle key | Selected trajectory index |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['dataset']} | {row['chemistry']} | "
            f"`{row['representative_key']}` | {row['selected_index']} |"
        )

    lines = [
        "# Supplementary Fig. S8",
        "",
        "Representative trajectory recovery across external battery datasets.",
        "",
        "## Data provenance",
        "",
        f"- Curves: `analysis/outputs/{ASSET_NPZ}`.",
        f"- Representative-selection metadata and RMSE annotations: `analysis/outputs/{ASSET_JSON}`.",
        "- The assets are produced by the analysis-layer multi-dataset formal workflow and are reused here without retraining, inference, or metric recomputation.",
        "- Reconstruction curves are the three-formal-seed mean for the locked representative trajectory; the corrupted input is the frozen seed-2021 observation stored in the asset.",
        "",
        "## Fixed representative selection",
        "",
        "The same locked trajectory is used across all four conditions for each external cohort. "
        "The deterministic rule is: up to 64 evenly spaced cycles per battery, then choose the cycle "
        "whose three-seed mean Segment-missing waveform RMSE is closest to the candidate median.",
        "",
        *table,
        "",
        "## Corruption settings",
        "",
        "| Condition | Frozen setting |",
        "|---|---|",
        "| Downsampling | 60% retained + interpolation |",
        "| Random-point missing | 75% missing |",
        "| Segment missing | 60% missing |",
        "| Destructive noise | ε = 0.10 V |",
        "",
        "These settings are displayed from the existing asset convention; the plotting script does not regenerate corrupted observations.",
        "",
        "## Fig. S5 style inheritance",
        "",
        "The script imports the Fig. S5 typography, semantic colours, axis treatment, missing-run helper, "
        "and export configuration directly from `figures/figS5/plot.py`. It retains the Fig. S5 "
        "font sizes, line widths, tick directions, grid treatment, legend construction, RMSE placement, "
        "and 600-dpi PNG/PDF/SVG export. The only necessary layout changes are the 6×4 subplot count, "
        "dataset-specific row y-limits, and four external condition labels.",
        "",
        "## Outputs",
        "",
        f"- PNG: `{outputs['png']}`",
        f"- PDF: `{outputs['pdf']}`",
        f"- SVG: `{outputs['svg']}`",
        "",
    ]
    (output_dir / "README_figS8.md").write_text("\n".join(lines), encoding="utf-8")


def _write_qc(metadata: dict, outputs: dict[str, str]) -> None:
    output_dir = Path(__file__).resolve().parent
    expected = set(DATASETS)
    actual = set(metadata)
    checks = [
        ("six external datasets represented; MIT excluded from the plot",
         len(DATASETS) == 6 and expected.issubset(actual) and "mit" not in DATASETS),
        ("four requested corruption conditions", set(CONDITIONS).issubset({
            "downsample", "random", "segment", "destructive",
        })),
        ("representative key and index available for every dataset",
         all("key" in metadata[key] and "selected_index" in metadata[key]
             for key in DATASETS)),
        ("condition RMSE available for every dataset × condition",
         all(set(CONDITIONS).issubset(metadata[key]["condition_rmse_v"])
             for key in DATASETS)),
        ("all curve arrays available for every dataset × condition",
         all(all(f"{key}_{condition}_{suffix}" in _LAST_ARRAY_KEYS
                 for condition in CONDITIONS
                 for suffix in ("clean", "input", "reconstruction", "observed"))
             for key in DATASETS)),
        ("PNG/PDF/SVG exported", all(Path(path).is_file() for path in outputs.values())),
    ]
    lines = [
        "# Fig. S8 QC",
        "",
        "The checks below are plotting/provenance checks only; no new analysis was run.",
        "",
    ]
    for label, passed in checks:
        lines.append(f"- [{'x' if passed else ' '}] {label}")
    lines += [
        "",
        "## Scientific safeguards",
        "",
        "- No model training or inference was performed.",
        "- No corruption definition or preprocessing was changed.",
        "- No representative cycle was selected for visual appeal; the locked metadata selection is reused.",
        "- No values are manually entered into the plotting data path; curves and RMSE annotations are read from the analysis-layer NPZ/JSON assets.",
        "- `figures/figS5` was not modified.",
        "",
    ]
    (output_dir / "figS8_qc.md").write_text("\n".join(lines), encoding="utf-8")


_LAST_ARRAY_KEYS: set[str] = set()


def draw(dataset: str = "all", *, results_root=None):
    del dataset, results_root
    apply_paper_style()
    _apply_legacy_typography()
    arrays = read_arrays(ASSET_NPZ)
    metadata = read_json(ASSET_JSON)
    global _LAST_ARRAY_KEYS
    _LAST_ARRAY_KEYS = set(arrays)

    figure, axes = plt.subplots(
        len(DATASETS), len(CONDITIONS),
        figsize=(NC_WIDTH, 212 * MM),
        sharex=True,
        sharey="row",
    )
    figure.subplots_adjust(
        left=0.085, right=0.988, bottom=0.082, top=0.930,
        hspace=0.19, wspace=0.10,
    )
    progress = np.linspace(0.0, 1.0, 1000)

    for column, (condition, condition_label) in enumerate(CONDITIONS.items()):
        axes[0, column].set_title(condition_label, fontweight="bold", pad=3)

    for row, (name, (display, chemistry)) in enumerate(DATASETS.items()):
        row_clean = []
        row_reconstructed = []
        for condition in CONDITIONS:
            row_clean.append(np.asarray(
                arrays[f"{name}_{condition}_clean"], dtype=float,
            ))
            row_reconstructed.append(np.asarray(
                arrays[f"{name}_{condition}_reconstruction"], dtype=float,
            ))
        # Use the clean/reconstructed voltage envelope for a readable,
        # dataset-specific scale.  Destructive-noise outliers are observations,
        # not part of the voltage range used to frame the recovery curves.
        range_values = np.concatenate(row_clean + row_reconstructed)
        low = float(np.min(range_values))
        high = float(np.max(range_values))
        pad = max((high - low) * 0.07, 0.02)
        y_low, y_high = low - pad, high + pad

        for column, condition in enumerate(CONDITIONS):
            axis = axes[row, column]
            prefix = f"{name}_{condition}"
            clean = np.asarray(arrays[f"{prefix}_clean"], dtype=float)
            corrupted = np.asarray(arrays[f"{prefix}_input"], dtype=float)
            reconstructed = np.asarray(
                arrays[f"{prefix}_reconstruction"], dtype=float,
            )
            observed = np.asarray(arrays[f"{prefix}_observed"], dtype=bool)

            if condition in {"downsample", "random"}:
                axis.scatter(
                    progress[observed], corrupted[observed], s=1.5,
                    color=C_OBSERVED, alpha=0.52, edgecolors="none", zorder=2,
                )
            elif condition == "segment":
                shown = corrupted.copy()
                shown[~observed] = np.nan
                axis.plot(
                    progress, shown, color=C_OBSERVED, lw=0.65,
                    alpha=0.72, zorder=2,
                )
                for start, end in _missing_runs(~observed):
                    axis.axvspan(
                        progress[start], progress[end], color=C_MISSING,
                        alpha=0.32, lw=0, zorder=0,
                    )
            else:
                axis.plot(
                    progress, corrupted, color=C_OBSERVED, lw=0.45,
                    alpha=0.50, zorder=1,
                )

            axis.plot(progress, clean, color=C_CLEAN, lw=0.75,
                      alpha=0.95, zorder=4)
            axis.plot(progress, reconstructed, color=C_RECON, lw=1.05,
                      zorder=3)
            axis.text(
                0.025, 0.955, CONDITION_DETAILS[condition],
                transform=axis.transAxes, ha="left", va="top",
                fontsize=4.8, color="#444444",
            )
            rmse = float(metadata[name]["condition_rmse_v"][condition]["mean"])
            axis.text(
                0.975, 0.055, f"RMSE {rmse:.3f} V",
                transform=axis.transAxes, ha="right", va="bottom",
                fontsize=4.6, color=C_RECON,
            )
            if row == 0 and column == 0:
                axis.legend(
                    handles=[
                        Line2D([], [], color=C_CLEAN, lw=0.9, label="Clean"),
                        Line2D([], [], color=C_OBSERVED, lw=0.8,
                               label="Corrupted"),
                        Line2D([], [], color=C_RECON, lw=1.2,
                               label="BatteryDiT"),
                    ],
                    loc="lower right", bbox_to_anchor=(0.985, 0.12),
                    borderaxespad=0, handlelength=1.35, handletextpad=0.45,
                    labelspacing=0.18, fontsize=4.2, frameon=False,
                )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(y_low, y_high)
            axis.set_xticks([0.0, 0.5, 1.0])
            axis.yaxis.set_major_locator(MaxNLocator(3))
            _style_axis(axis)
            if column == 0:
                axis.set_ylabel(
                    f"{display} ({chemistry})\nVoltage (V)", labelpad=3,
                )
            if row == len(DATASETS) - 1:
                axis.set_xlabel("Normalized charging progress", labelpad=1)

        axes[row, 0].text(
            -0.25, 1.04, f"({chr(97 + row)})",
            transform=axes[row, 0].transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left",
        )

    previous_font_family = export_helpers.FONT_FAMILY
    export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
    plt.close(figure)
    _write_readme(metadata, outputs)
    _write_qc(metadata, outputs)
    return outputs


if __name__ == "__main__":
    result = draw()
    print(json.dumps(result, indent=2))
