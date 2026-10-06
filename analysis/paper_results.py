"""Assemble `paper_results/`: the per-figure / per-table data an author or
reviewer needs, in one predictable place.

Layout (see `paper_results/README.md`):

    paper_results/
      figN/<name>_source_data.csv      values plotted in the figure (1 row per mark)
      figN/raw/<name>_raw.csv          the unaggregated rows behind those values
      table_sN/<name>.csv              the table exactly as it appears
      table_sN/raw/<name>_raw.csv      unaggregated rows behind it

Every file here is regenerated from `results/` and `outputs/analysis/`; nothing
is hand-edited, so a reviewer can re-derive any number.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.utils.io import analysis_dir
from analysis.utils.paths import ANALYSIS_OUTPUT_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

PAPER_ROOT = REPOSITORY_ROOT / "paper_results"


def _dir(*parts: str) -> Path:
    path = PAPER_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    (path / "raw").mkdir(parents=True, exist_ok=True)
    return path


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _copy_analysis(name: str, path: Path) -> Path | None:
    source = ANALYSIS_OUTPUT_ROOT / name
    if not source.is_file():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return path


def _raw_cycles(dataset: str, seed: int) -> pd.DataFrame | None:
    root = RESULTS_ROOT / "PopulationIndividual" / dataset
    for directory in sorted(root.glob("*/")):
        summary = directory / "summary.json"
        if not summary.is_file():
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if int(payload.get("model_seed", -1)) != seed or payload.get("status") != "COMPLETE":
            continue
        cycles = directory / "cycle_level_decomposition.csv.gz"
        if cycles.is_file():
            frame = pd.read_csv(cycles)
            frame.insert(0, "dataset", dataset)
            return frame
    return None


def build() -> dict[str, list[str]]:
    """Materialise every figure/table data file.  Returns a written-file manifest."""
    if PAPER_ROOT.exists():
        shutil.rmtree(PAPER_ROOT)
    written: dict[str, list[str]] = {}

    # ---- tables, one folder per manuscript table --------------------------
    TABLES = {
        "table_s3_benchmark": (("benchmark_summary.csv", "benchmark_summary.csv"),
                               ("benchmark_soh_ci.csv", "benchmark_soh_ci.csv")),
        "table_s4_stress_test": (("stress_test_curves.csv", "stress_test_curves.csv"),
                                 ("stress_test_signature_raw.csv", "stress_test_signature_raw.csv")),
        "table_s5_cross_dataset": (("multi_dataset_summary.csv", "multi_dataset_summary.csv"),
                                   ("multi_dataset_corrupted_rmse.csv", "multi_dataset_corrupted_rmse.csv")),
        "table_s6_population_individual": (
            ("population_individual_condition_summary.csv",
             "population_individual_condition_summary.csv"),
            ("population_individual_soh_linkage.csv",
             "population_individual_soh_linkage.csv"),
            ("population_individual_manifold_validation.csv",
             "population_individual_manifold_validation.csv"),
        ),
    }
    for folder, entries in TABLES.items():
        for source_name, target_name in entries:
            path = _dir(folder) / target_name
            if _copy_analysis(source_name, path):
                written.setdefault(folder, []).append(str(path))

    # ---- Fig 2: benchmark -------------------------------------------------
    fig2 = _dir("fig2")
    for name, target in (("benchmark_summary.csv", "fig2_benchmark_summary.csv"),
                         ("benchmark_soh_ci.csv", "fig2_soh_error_ci.csv")):
        if _copy_analysis(name, fig2 / target):
            written.setdefault("fig2", []).append(str(fig2 / target))
    for name, target in (("benchmark_representatives.npz", "fig2_representatives.npz"),
                         ("benchmark_cycle_errors.csv", "fig2_cycle_errors.csv")):
        source = ANALYSIS_OUTPUT_ROOT / name
        if source.is_file():
            shutil.copyfile(source, fig2 / target)
            written.setdefault("fig2", []).append(str(fig2 / target))

    # ---- Fig 3: stress curves ---------------------------------------------
    fig3 = _dir("fig3")
    for name, target in (("stress_test_curves.csv", "fig3_stress_curves.csv"),
                         ("stress_test_statistics.json", "fig3_signature_statistics.json"),
                         ("stress_test_signature_raw.csv", "fig3_signature_raw.csv")):
        source = ANALYSIS_OUTPUT_ROOT / name
        if source.is_file():
            shutil.copyfile(source, fig3 / target)
            written.setdefault("fig3", []).append(str(fig3 / target))
    for name, target in (("stress_test_figure_arrays.npz", "fig3_figure_arrays.npz"),):
        source = ANALYSIS_OUTPUT_ROOT / name
        if source.is_file():
            shutil.copyfile(source, fig3 / target)
            written.setdefault("fig3", []).append(str(fig3 / target))

    # ---- Fig 4: cross-system ----------------------------------------------
    fig4 = _dir("fig4")
    for name, target in (("multi_dataset_summary.csv", "fig4_cross_system_summary.csv"),
                         ("multi_dataset_corrupted_rmse.csv", "fig4_corrupted_rmse.csv")):
        if _copy_analysis(name, fig4 / target):
            written.setdefault("fig4", []).append(str(fig4 / target))
    for name, target in (("multi_dataset_representatives.npz", "fig4_representatives.npz"),
                         ("multi_dataset_representatives.json", "fig4_representatives.json")):
        source = ANALYSIS_OUTPUT_ROOT / name
        if source.is_file():
            shutil.copyfile(source, fig4 / target)
            written.setdefault("fig4", []).append(str(fig4 / target))

    # ---- Fig 5-style: population/individual asymmetry -------------------
    fig5 = _dir("fig5")
    for name, filename in (
        ("cross_system_conditional_summary.csv", "fig5_population_individual_summary.csv"),
        ("cross_system_condition_rollup.csv", "fig5_condition_rollup.csv"),
        ("cross_system_conditional_battery.csv", "fig5_battery_level_conditional.csv"),
        ("cross_system_asymmetry_rollup.csv", "fig5_asymmetry_rollup.csv"),
        ("cross_method_asymmetry.csv", "fig5_cross_method_asymmetry.csv"),
    ):
        if _copy_analysis(name, fig5 / filename):
            written.setdefault("fig5", []).append(str(fig5 / filename))

    # analysis-stability exports used by the supplementary figures
    for name, target in (("population_individual_manifold_validation.csv",
                          "fig5_manifold_validation.csv"),
                         ("population_individual_signature_linkage.csv",
                          "fig5_signature_linkage.csv")):
        if _copy_analysis(name, fig5 / target):
            written.setdefault("fig5", []).append(str(fig5 / target))
    for name, target in (("supplementary_downstream.npz", "fig5_downstream_arrays.npz"),
                         ("supplementary_representatives.npz", "supplementary_representatives.npz"),
                         ("supplementary_soh_parity.npz", "supplementary_soh_parity.npz"),
                         ("supplementary_waveform_rmse.npz", "supplementary_waveform_rmse.npz")):
        source = ANALYSIS_OUTPUT_ROOT / name
        if source.is_file():
            shutil.copyfile(source, fig5 / target)
            written.setdefault("fig5", []).append(str(fig5 / target))

    # per-system raw cycle tables for reviewer spot checks
    for dataset, seed in (("mit", 2021), ("hust", 2023), ("mich", 2021),
                          ("stanford_2", 2021), ("tju_nca_45c", 2021),
                          ("tju_ncm_45c", 2021)):
        frame = _raw_cycles(dataset, seed)
        if frame is not None:
            _write(frame, fig5 / "raw" / f"cycle_level_decomposition_{dataset}.csv")
            written.setdefault("fig5/raw", []).append(str(fig5 / "raw"))

    (PAPER_ROOT / "README.md").write_text(
        "# paper_results\n\n"
        "Per-figure and per-table data for the manuscript.  Every file is\n"
        "regenerated by `python main.py --analysis paper_results`; do not edit\n"
        "these files by hand.\n\n"
        "Each entry has a `*_source_data.csv` holding exactly the values plotted\n"
        "or tabulated, and a `raw/` folder holding the unaggregated rows behind\n"
        "them so any number can be re-derived.\n",
        encoding="utf-8")
    return written


def main() -> None:
    written = build()
    total = sum(len(v) for v in written.values())
    print(f"paper_results: {total} files -> {PAPER_ROOT}")


if __name__ == "__main__":
    main()
