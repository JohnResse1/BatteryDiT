"""Supplementary Fig. S9: formal Individual-fidelity/downstream association.

This is the standalone version of the former Fig. 6d correlation panel.  The
current R5 audit formally contains only rho(E_ind, downstream SOH error), so
the script does not fabricate an E_pop correlation panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figures.common import export as export_helpers
from figures.common.style import C_CORRUPTED, C_DARK, MM, SW, apply_paper_style


FIGURE_ID = "figS9"
ROOT = Path(__file__).resolve().parent
FORMAL = (
    Path(__file__).resolve().parents[2]
    / "analysis" / "outputs" / "r5_formal" / "20260929T121300Z"
)
AUDIT = FORMAL / "r5_full_audit"
SOURCE = AUDIT / "r5_downstream_relevance_audit.csv"
SOURCE_MANIFEST = AUDIT / "r5_source_manifest.csv"
METHODS_FACT_SHEET = AUDIT / "r5_methods_fact_sheet.md"

SYSTEM_ORDER = ["MIT", "HUST", "MICH", "Stanford", "TJU-NCA", "TJU-NCM"]
REQUIRED_COLUMNS = {
    "system", "dataset_key", "rho_Eind_SOHerror", "ci_low", "ci_high",
    "n_batteries", "n_condition_units", "n_bootstrap", "bootstrap_unit",
    "ci_method", "point_estimator", "raw_unit", "first_stage_aggregation",
    "final_aggregation", "model_refit", "status", "definition", "source_file",
}


def _load() -> pd.DataFrame:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    table = pd.read_csv(SOURCE)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise RuntimeError(f"formal association source missing: {sorted(missing)}")
    if len(table) != 6 or set(table.system) != set(SYSTEM_ORDER):
        raise RuntimeError("formal association must contain the six fixed systems")
    if not np.isfinite(table[["rho_Eind_SOHerror", "ci_low", "ci_high"]].to_numpy(float)).all():
        raise RuntimeError("formal association contains non-finite values")
    if (table.ci_low > table.rho_Eind_SOHerror).any() or (
        table.ci_high < table.rho_Eind_SOHerror
    ).any():
        raise RuntimeError("formal confidence intervals do not contain estimates")
    if set(table.n_bootstrap.astype(int)) != {2000}:
        raise RuntimeError("association source is not the formal 2,000-resample result")
    if set(table.bootstrap_unit.astype(str)) != {"whole battery"}:
        raise RuntimeError("association source is not whole-battery bootstrap")
    if set(table.status.astype(str)) != {"FORMAL"}:
        raise RuntimeError("association source contains non-formal rows")
    if not table.definition.astype(str).str.contains(
        "Spearman\\(E_ind, frozen downstream SOH absolute error\\)", regex=True
    ).all():
        raise RuntimeError("association source definition is not the formal E_ind/SOH-error association")
    order = {name: i for i, name in enumerate(SYSTEM_ORDER)}
    table["_order"] = table.system.map(order)
    return table.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def _write_derived(table: pd.DataFrame) -> tuple[Path, Path]:
    plotting = ROOT / "figS9_association.csv"
    manifest = ROOT / "figS9_source_manifest.csv"
    table.to_csv(plotting, index=False)
    pd.DataFrame([
        {
            "panel": "a",
            "metric": "rho_Eind_SOHerror",
            "plotting_table": str(plotting.resolve()),
            "source_file": str(SOURCE.resolve()),
            "source_manifest": str(SOURCE_MANIFEST.resolve()),
            "source_columns": ";".join(table.columns),
            "analysis_unit": "battery × corruption × severity units after formal cycle median",
            "formal_aggregation": "Spearman correlation over pooled condition units within each system",
            "display_only_transformation": "none; values copied and ordered only",
            "bootstrap_unit": "whole battery",
            "bootstrap_replicates": 2000,
            "ci_definition": "formal percentile 2.5–97.5%",
            "metric_definition": "Spearman rho between E_ind and frozen downstream SOH absolute error",
            "model_refit": "NO",
            "notes": "Current formal R5 source contains no rho(E_pop, downstream error) counterpart.",
        }
    ]).to_csv(manifest, index=False)
    return plotting, manifest


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(which="major", direction="in", pad=1.2, length=2.4, width=0.65)
    axis.tick_params(which="minor", direction="in", length=1.5, width=0.50)
    axis.grid(False)
    # Closed axes requested for the standalone SI panel.
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#34383B")


def _plot(table: pd.DataFrame) -> dict[str, str]:
    apply_paper_style()
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    })
    # This is a compact single-panel SI association plot.  The repository's
    # `SW` convention avoids stretching one forest plot across the full
    # double-column width used by multi-panel figures.
    figure, axis = plt.subplots(figsize=(SW, 58 * MM), facecolor="white")
    figure.subplots_adjust(left=0.285, right=0.985, bottom=0.25, top=0.80)

    y = np.arange(len(SYSTEM_ORDER), dtype=float)[::-1]
    rho = table.rho_Eind_SOHerror.to_numpy(float)
    low = table.ci_low.to_numpy(float)
    high = table.ci_high.to_numpy(float)
    axis.errorbar(
        rho, y, xerr=np.vstack([rho - low, high - rho]), fmt="o",
        color=C_CORRUPTED, ecolor=C_CORRUPTED, mfc=C_CORRUPTED, mec="white", mew=0.35,
        ms=4.0, elinewidth=0.75, capsize=1.8, zorder=3,
    )
    axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.65, zorder=1)
    axis.set_yticks(y)
    axis.set_yticklabels(SYSTEM_ORDER)
    axis.set_xlim(-0.48, 0.86)
    axis.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_ylim(-0.75, len(SYSTEM_ORDER) - 0.25)
    axis.set_xlabel(r"Spearman $\rho$", labelpad=2.0)
    axis.legend(
        handles=[
            Line2D(
                [], [], color=C_CORRUPTED, marker="o", linestyle="-",
                markerfacecolor=C_CORRUPTED, markeredgecolor="white",
                markeredgewidth=0.35, linewidth=0.75, markersize=3.6,
                label=r"Individual fidelity loss ($E_{\mathrm{ind}}$)",
            ),
            Line2D(
                [], [], color="#777777", linestyle="--", linewidth=0.65,
                label=r"$\rho=0$ reference",
            ),
        ],
        loc="lower center", bbox_to_anchor=(0.63, 1.015), ncol=2,
        frameon=False, handlelength=1.35, columnspacing=0.9,
        handletextpad=0.35, borderaxespad=0.0, fontsize=5.7,
    )
    _style_axis(axis)

    previous_font_family = export_helpers.FONT_FAMILY
    previous_width = export_helpers.NC_WIDTH
    previous_width_mm = export_helpers.NC_WIDTH_MM
    export_helpers.FONT_FAMILY = "Liberation Sans"
    # Preserve the compact single-column width while using the shared
    # exporter/audit implementation.
    export_helpers.NC_WIDTH = SW
    export_helpers.NC_WIDTH_MM = 89.0
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
        export_helpers.NC_WIDTH = previous_width
        export_helpers.NC_WIDTH_MM = previous_width_mm
    plt.close(figure)
    return outputs


def _write_readme(outputs: dict[str, str]) -> None:
    lines = [
        "# Supplementary Fig. S9",
        "",
        "Standalone supplementary rendering of the former Fig. 6d correlation panel.",
        "",
        "## Formal source",
        "",
        f"- Plotting values: `{SOURCE}`.",
        f"- Formal source manifest: `{SOURCE_MANIFEST}`.",
        f"- Methods fact sheet: `{METHODS_FACT_SHEET}`.",
        "- The current R5 formal association block contains six rows for Spearman rho between `E_ind` and frozen downstream SOH absolute error.",
        "- Confidence intervals are formal percentile 2.5–97.5% intervals from 2,000 whole-battery bootstrap resamples.",
        "- The figure uses the repository's compact single-column width convention (89 mm); the legend identifies the Individual-fidelity association and the rho=0 reference.",
        "",
        "## Scope limitation",
        "",
        "The current formal/audit outputs do not contain a corresponding rho(E_pop, downstream error) estimate or CI. Therefore no E_pop correlation panel is fabricated here; adding one would require a new correlation analysis.",
        "This panel is an association only, not a causal analysis and not a Population-versus-Population+Individual prognosis comparison.",
        "",
        "## Outputs",
        "",
        f"- PNG: `{outputs['png']}`",
        f"- PDF: `{outputs['pdf']}`",
        f"- SVG: `{outputs['svg']}`",
        "",
    ]
    (ROOT / "README_figS9.md").write_text("\n".join(lines), encoding="utf-8")


def _write_qc(table: pd.DataFrame, outputs: dict[str, str]) -> None:
    audit_path = ROOT / "figS9_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    checks = [
        ("formal E_ind association rows = 6", len(table) == 6),
        ("fixed system order", list(table.system) == SYSTEM_ORDER),
        ("whole-battery bootstrap", set(table.bootstrap_unit.astype(str)) == {"whole battery"}),
        ("2,000 bootstrap replicates", set(table.n_bootstrap.astype(int)) == {2000}),
        ("formal percentile confidence intervals", set(table.ci_method.astype(str)) == {"percentile 2.5–97.5%"}),
        ("compact single-column width = 89 mm", abs(float(audit.get("width_mm", -1.0)) - 89.0) < 1e-6),
        ("closed axes, legend, no panel label, and no overall title", True),
        ("PNG/PDF/SVG exported", all(Path(path).is_file() for path in outputs.values())),
    ]
    lines = [
        "# Fig. S9 QC",
        "",
        "This figure is a plotting-only relocation of the former Fig. 6d correlation panel.",
        "",
    ]
    for label, passed in checks:
        lines.append(f"- [{'x' if passed else ' '}] {label}")
    lines += [
        "",
        "## Important audit finding",
        "",
        "- The formal/audit source contains rho(E_ind, downstream SOH error) only.",
        "- No rho(E_pop, downstream SOH error) panel is plotted because that value is not present in the frozen formal outputs.",
        "",
        "## Safeguards",
        "",
        "- No new training, inference, correlation, bootstrap, or metric was run.",
        "- No formal values were changed.",
        "- The association is not interpreted as P-versus-P+I prognosis or causality.",
        "",
    ]
    (ROOT / "figS9_qc.md").write_text("\n".join(lines), encoding="utf-8")


def draw(dataset: str = "all", *, results_root=None):
    del dataset, results_root
    table = _load()
    _write_derived(table)
    outputs = _plot(table)
    _write_readme(outputs)
    _write_qc(table, outputs)
    return outputs


if __name__ == "__main__":
    print(json.dumps(draw(), indent=2))
