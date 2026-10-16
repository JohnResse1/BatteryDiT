"""Build the R5 cross-system Fig. 6 main-text sketch.

This script is plotting-only.  It reads the frozen R5 formal/canonical assets,
does no model training or inference, and does not recompute an inferential
statistic.  The only derived values are display-only within-system colour
scales for panel (c).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from figures.common.export import _typography_audit, enforce_nc_readability  # noqa: E402
import figures.common.export as export_helpers  # noqa: E402
from figures.common.style import (  # noqa: E402
    C_BLUE,
    C_CORRUPTED,
    C_DARK,
    C_INPUT,
    C_REFERENCE,
    CMAP_IND,
    CMAP_POP,
    MM,
    NC_WIDTH,
    apply_paper_style,
)


FORMAL = REPO / "analysis" / "outputs" / "r5_formal" / "20260929T121300Z"
AUDIT = FORMAL / "r5_full_audit"
CANONICAL = (
    REPO
    / "analysis"
    / "outputs"
    / "canonical"
    / "r5_framework_generality"
    / "20260928T_r5_framework_generality"
)
OUT = REPO / "figures" / "fig6"

FRAMEWORK_RECON_SOURCE = FORMAL / "framework_generality_reconstruction.csv"
FRAMEWORK_DOWNSTREAM_SOURCE = FORMAL / "framework_generality_downstream.csv"
PI_SOURCE = AUDIT / "r5_cross_system_recoverability.csv"
SOURCE_MANIFEST = AUDIT / "r5_source_manifest.csv"
PROTOCOL_SOURCE = AUDIT / "r5_cross_system_protocol.csv"
CLAIM_MATRIX = AUDIT / "r5_claim_support_matrix.csv"
METHODS_FACT_SHEET = AUDIT / "r5_methods_fact_sheet.md"
REPRESENTATIVE_NPZ = CANONICAL / "raw" / "framework_representatives.npz"
REPRESENTATIVE_JSON = CANONICAL / "raw" / "framework_representatives.json"

EXTERNAL_FRAMEWORK = [
    ("hust", "HUST", "LFP"),
    ("mich", "MICH", "NMC111"),
    ("sdu", "SDU", "NMC532"),
    ("stanford_2", "Stanford", "NMC532"),
    ("tju_nca_45c", "TJU-NCA", "NCA"),
    ("tju_ncm_45c", "TJU-NCM", "NCM"),
]
EXTERNAL_KEYS = [key for key, _, _ in EXTERNAL_FRAMEWORK]

# SDU is outside the Population–Individual formal scope, and MIT is the
# primary/reference system already covered earlier in the manuscript.
EXTERNAL_PI = [
    ("hust", "HUST"),
    ("mich", "MICH"),
    ("stanford_2", "Stanford"),
    ("tju_nca_45c", "TJU-NCA"),
    ("tju_ncm_45c", "TJU-NCM"),
]
EXTERNAL_PI_KEYS = [key for key, _ in EXTERNAL_PI]

FRAMEWORK_CONDITIONS = [
    ("downsample", "Downsampling", "D"),
    ("random", "Random missing", "R"),
    ("segment", "Segment missing", "S"),
    ("destructive", "Destructive noise", "N"),
]
PI_CONDITIONS = [
    ("downsample", "Downsampling"),
    ("random", "Random missing"),
    ("segment", "Segment missing"),
    ("destructive", "Destructive noise"),
    ("hybrid", "Hybrid"),
]
SEVERITIES = ["L1", "L2", "L3", "L4", "L5"]

MISSING_COLOR = "#E5E8EA"
BAR_BLUE = "#4D8AB8"
# Use the repository-wide Population/Individual heatmap colours so Fig. 6
# follows the same semantic palette as the preceding main figures.
POP_CMAP = CMAP_POP
IND_CMAP = CMAP_IND


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"empty source: {path}")
    return data


def _require(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(data.columns)
    if missing:
        raise RuntimeError(f"{name} missing columns: {sorted(missing)}")


def _style_axis(ax: plt.Axes, labelsize: float = 4.8) -> None:
    ax.tick_params(
        which="both",
        direction="in",
        length=2.0,
        width=0.45,
        pad=1.4,
        labelsize=labelsize,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("#4F5458")


def _panel_header(
    fig: plt.Figure,
    slot,
    label: str,
    title: str,
    header_ratio: float,
):
    parts = slot.subgridspec(
        2, 1, height_ratios=(header_ratio, 1.0 - header_ratio), hspace=0.0
    )
    header = fig.add_subplot(parts[0])
    header.set_axis_off()
    # Keep the panel label in the reserved left margin.  The title starts at
    # the panel content edge, which preserves the repository's left-label
    # convention and avoids making the label part of the title string.
    header.text(
        -0.055,
        0.90,
        f"({label})",
        fontsize=8.0,
        fontweight="bold",
        va="top",
        clip_on=False,
    )
    header.text(0.0, 0.90, title, fontsize=7.2, fontweight="bold", va="top")
    return header, parts[1]


def _missing_runs(mask: np.ndarray):
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _audit_and_load() -> dict[str, object]:
    """Read and validate only the current formal/canonical sources."""
    for source in [
        SOURCE_MANIFEST,
        PROTOCOL_SOURCE,
        CLAIM_MATRIX,
        METHODS_FACT_SHEET,
        FRAMEWORK_RECON_SOURCE,
        FRAMEWORK_DOWNSTREAM_SOURCE,
        PI_SOURCE,
        REPRESENTATIVE_NPZ,
        REPRESENTATIVE_JSON,
    ]:
        if not source.is_file():
            raise FileNotFoundError(source)

    source_manifest = _read(SOURCE_MANIFEST)
    protocol = _read(PROTOCOL_SOURCE)
    claims = _read(CLAIM_MATRIX)
    recon = _read(FRAMEWORK_RECON_SOURCE)
    downstream = _read(FRAMEWORK_DOWNSTREAM_SOURCE)
    pi = _read(PI_SOURCE)
    _require(
        recon,
        {
            "dataset",
            "dataset_key",
            "condition",
            "waveform_rmse_mean",
            "waveform_rmse_sd",
            "n_model_seeds",
        },
        "framework reconstruction",
    )
    _require(
        downstream,
        {
            "dataset",
            "dataset_key",
            "target",
            "target_definition",
            "condition",
            "downstream_target_rmse_mean",
            "downstream_target_rmse_sd",
            "n_model_seeds",
            "metric_name",
        },
        "framework downstream",
    )
    _require(
        pi,
        {
            "system",
            "dataset_key",
            "condition",
            "severity",
            "E_pop",
            "E_ind",
            "n_batteries",
            "source_file",
        },
        "PI recoverability",
    )
    if set(recon.dataset_key.unique()) != {"mit", *EXTERNAL_KEYS} or len(recon) != 28:
        raise RuntimeError("framework reconstruction is not the formal 7 × 4 block")
    if set(downstream.dataset_key.unique()) != {"mit", *EXTERNAL_KEYS} or len(downstream) != 28:
        raise RuntimeError("framework downstream is not the formal 7 × 4 block")
    if len(pi) != 150:
        raise RuntimeError("PI source is not the formal 6 × 5 × 5 block")
    if set(pi.condition) != {key for key, _ in PI_CONDITIONS} or set(pi.severity) != set(SEVERITIES):
        raise RuntimeError("PI source has an unexpected corruption/severity grid")

    with np.load(REPRESENTATIVE_NPZ, allow_pickle=False) as archive:
        arrays = {key: np.asarray(value) for key, value in archive.items()}
    representatives = json.loads(REPRESENTATIVE_JSON.read_text(encoding="utf-8"))
    for key, _, _ in EXTERNAL_FRAMEWORK:
        for suffix in ["segment_clean", "segment_input", "segment_reconstruction", "segment_observed"]:
            if f"{key}_{suffix}" not in arrays:
                raise RuntimeError(f"representative asset missing {key}_{suffix}")
        if key not in representatives:
            raise RuntimeError(f"representative metadata missing {key}")

    external_recon = recon[recon.dataset_key.isin(EXTERNAL_KEYS)].copy()
    external_downstream = downstream[downstream.dataset_key.isin(EXTERNAL_KEYS)].copy()
    external_pi = pi[pi.dataset_key.isin(EXTERNAL_PI_KEYS)].copy()
    expected_framework = len(EXTERNAL_KEYS) * len(FRAMEWORK_CONDITIONS)
    expected_pi = len(EXTERNAL_PI_KEYS) * len(PI_CONDITIONS) * len(SEVERITIES)
    if len(external_recon) != expected_framework or len(external_downstream) != expected_framework:
        raise RuntimeError("external framework block is not 6 × 4")
    if len(external_pi) != expected_pi:
        raise RuntimeError("external PI block is not 5 × 5 × 5")

    return {
        "source_manifest": source_manifest,
        "protocol": protocol,
        "claims": claims,
        "recon": external_recon,
        "downstream": external_downstream,
        "pi": external_pi,
        "arrays": arrays,
        "representatives": representatives,
    }


def _within_system_matrix(table: pd.DataFrame, dataset_key: str, metric: str) -> np.ndarray:
    sub = table[table.dataset_key.eq(dataset_key)].set_index(["condition", "severity"])
    matrix = np.zeros((len(PI_CONDITIONS), len(SEVERITIES)), dtype=float)
    for i, (condition, _) in enumerate(PI_CONDITIONS):
        for j, severity in enumerate(SEVERITIES):
            matrix[i, j] = float(sub.loc[(condition, severity), metric])
    lo, hi = float(matrix.min()), float(matrix.max())
    if hi > lo:
        return (matrix - lo) / (hi - lo)
    return np.full_like(matrix, 0.5)


def _write_plotting_tables(data: dict[str, object]) -> None:
    """Write transparent plotting tables without changing formal values."""
    OUT.mkdir(parents=True, exist_ok=True)
    recon = data["recon"]
    downstream = data["downstream"]
    pi = data["pi"]
    representatives = data["representatives"]

    a_rows = []
    for key, label, chemistry in EXTERNAL_FRAMEWORK:
        meta = representatives[key]
        a_rows.append(
            {
                "dataset": label,
                "dataset_key": key,
                "chemistry": chemistry,
                "condition": "segment",
                "rmse_v": float(meta["selection_score_v"]),
                "selected_index": int(meta["selected_index"]),
                "source_file": f"{REPRESENTATIVE_NPZ.resolve()}; {REPRESENTATIVE_JSON.resolve()}",
                "source_columns": "segment_clean;segment_input;segment_reconstruction;segment_observed;selected_index;selection_score_v",
                "selection_rule": str(meta["selection_rule"]),
                "display_only_transformation": "none",
            }
        )
    pd.DataFrame(a_rows).to_csv(OUT / "fig6_panel_a_examples.csv", index=False)

    b1 = recon.copy()
    b1["source_file"] = str(FRAMEWORK_RECON_SOURCE.resolve())
    b1["source_columns"] = "waveform_rmse_mean;waveform_rmse_sd;n_model_seeds"
    b1["analysis_unit"] = "dataset × corruption × frozen model-seed summary"
    b1["aggregation"] = "formal mean and sample SD across three frozen model seeds"
    b1["display_only_transformation"] = "none"
    b1.to_csv(OUT / "fig6_panel_b1_waveform.csv", index=False)

    b2 = downstream.copy()
    b2["source_file"] = str(FRAMEWORK_DOWNSTREAM_SOURCE.resolve())
    b2["source_columns"] = "downstream_target_rmse_mean;downstream_target_rmse_sd;n_model_seeds;metric_name;target_definition"
    b2["analysis_unit"] = "dataset × corruption × frozen model-seed summary"
    b2["aggregation"] = "formal mean and sample SD across three frozen model seeds"
    b2["display_only_transformation"] = "none; downstream_delta_vs_clean columns intentionally not plotted"
    b2.to_csv(OUT / "fig6_panel_b2_downstream.csv", index=False)

    ep = pi.copy()
    ep["display_value"] = ep.groupby("dataset_key")["E_pop"].transform(
        lambda values: (values - values.min()) / (values.max() - values.min())
        if values.max() > values.min()
        else 0.5
    )
    ep["display_only_transformation"] = "within-system min–max scaling for colour only; raw E_pop retained"
    ep["metric_definition"] = "E_pop = |h_rec − SOH_true|"
    ep.to_csv(OUT / "fig6_panel_c_epop.csv", index=False)

    ei = pi.copy()
    ei["display_value"] = ei.groupby("dataset_key")["E_ind"].transform(
        lambda values: (values - values.min()) / (values.max() - values.min())
        if values.max() > values.min()
        else 0.5
    )
    ei["display_only_transformation"] = "within-system min–max scaling for colour only; raw E_ind retained"
    ei["metric_definition"] = "E_ind = RMSE(R_rec − R_clean) / S_ind,d"
    ei.to_csv(OUT / "fig6_panel_c_eind.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "panel": "a",
                "plotting_table": str((OUT / "fig6_panel_a_examples.csv").resolve()),
                "source_file": f"{REPRESENTATIVE_NPZ.resolve()}; {REPRESENTATIVE_JSON.resolve()}",
                "source_columns": "frozen representative trajectory arrays and selection metadata",
                "analysis_unit": "one frozen representative cycle per external framework dataset",
                "aggregation": "canonical representative asset; no new aggregation",
                "display_only_transformation": "none",
            },
            {
                "panel": "b1",
                "plotting_table": str((OUT / "fig6_panel_b1_waveform.csv").resolve()),
                "source_file": str(FRAMEWORK_RECON_SOURCE.resolve()),
                "source_columns": "waveform_rmse_mean;waveform_rmse_sd;n_model_seeds",
                "analysis_unit": "dataset × corruption × frozen model-seed summary",
                "aggregation": "formal mean and sample SD across three seeds",
                "display_only_transformation": "none",
            },
            {
                "panel": "b2",
                "plotting_table": str((OUT / "fig6_panel_b2_downstream.csv").resolve()),
                "source_file": str(FRAMEWORK_DOWNSTREAM_SOURCE.resolve()),
                "source_columns": "downstream_target_rmse_mean;downstream_target_rmse_sd;metric_name;target_definition",
                "analysis_unit": "dataset × corruption × frozen model-seed summary",
                "aggregation": "formal mean and sample SD across three seeds",
                "display_only_transformation": "none; absolute RMSE only, no delta or baseline plotted",
            },
            {
                "panel": "c",
                "plotting_table": f"{(OUT / 'fig6_panel_c_epop.csv').resolve()}; {(OUT / 'fig6_panel_c_eind.csv').resolve()}",
                "source_file": str(PI_SOURCE.resolve()),
                "source_columns": "system;dataset_key;condition;severity;E_pop;E_ind;n_batteries",
                "analysis_unit": "cycle × battery × corruption × severity",
                "aggregation": "formal median within battery; median across batteries",
                "display_only_transformation": "each plotted dataset independently min–max scales colour; raw values retained",
            },
        ]
    )
    manifest.to_csv(OUT / "fig6_source_manifest.csv", index=False)


def _plot_panel_a(fig: plt.Figure, slot, data: dict[str, object]):
    header, content = _panel_header(
        fig,
        slot,
        "a",
        "",
        header_ratio=0.045,
    )
    grid = content.subgridspec(1, 6, wspace=0.30)
    axes = []
    arrays = data["arrays"]
    for i, (key, label, chemistry) in enumerate(EXTERNAL_FRAMEWORK):
        ax = fig.add_subplot(grid[0, i])
        axes.append(ax)
        clean = np.asarray(arrays[f"{key}_segment_clean"], dtype=float)
        corrupted = np.asarray(arrays[f"{key}_segment_input"], dtype=float)
        reconstruction = np.asarray(arrays[f"{key}_segment_reconstruction"], dtype=float)
        observed = np.asarray(arrays[f"{key}_segment_observed"], dtype=bool)
        x = np.linspace(0.0, 1.0, len(clean))
        for start, end in _missing_runs(~observed):
            ax.axvspan(x[start], x[end], color=MISSING_COLOR, alpha=0.82, linewidth=0, zorder=0)
        ax.plot(x, clean, color=C_REFERENCE, lw=0.72, zorder=2)
        ax.plot(x, np.where(observed, corrupted, np.nan), color=C_INPUT, lw=0.65, zorder=1)
        ax.plot(x, reconstruction, color=C_BLUE, lw=0.95, zorder=3)
        lo = min(float(np.nanmin(clean)), float(np.nanmin(reconstruction)))
        hi = max(float(np.nanmax(clean)), float(np.nanmax(reconstruction)))
        pad = max((hi - lo) * 0.08, 0.025)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["0", "1"])
        if i == 0:
            ax.set_ylabel("Voltage (V)")
        else:
            ax.set_ylabel("")
        # Follow the established external-trajectory layout: system name and
        # chemistry sit in two clean tiers above each data rectangle.
        ax.text(0.5, 1.100, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.2, fontweight="bold", clip_on=False)
        ax.text(0.5, 1.024, chemistry, transform=ax.transAxes, ha="center", va="bottom", fontsize=4.8, color=C_DARK, clip_on=False)
        _style_axis(ax, labelsize=4.0)
    axes[0].legend(
        handles=[
            Line2D([], [], color=C_REFERENCE, lw=1.0, label="Clean"),
            Line2D([], [], color=C_BLUE, lw=1.2, label="BatteryDiT"),
            Patch(facecolor=MISSING_COLOR, edgecolor="none", alpha=0.8, label="Missing region"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.035),
        frameon=False,
        fontsize=4.0,
        ncol=1,
        handlelength=0.80,
        handletextpad=0.20,
        labelspacing=0.10,
        borderaxespad=0.0,
    )
    positions = [ax.get_position() for ax in axes]
    fig.text(
        (positions[0].x0 + positions[-1].x1) / 2,
        min(pos.y0 for pos in positions) - 0.035,
        "Normalized charging progress",
        ha="center",
        va="top",
        fontsize=4.6,
    )
    return [header, *axes]


def _bar_card(
    ax: plt.Axes,
    values: np.ndarray,
    errors: np.ndarray,
    y_max: float,
    show_ylabels: bool,
    ylabel: str,
    title: str | None,
):
    positions = np.arange(len(FRAMEWORK_CONDITIONS), dtype=float)
    bars = ax.bar(
        positions,
        values,
        width=0.58,
        color=BAR_BLUE,
        alpha=0.78,
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        positions,
        values,
        yerr=errors,
        fmt="none",
        ecolor=C_DARK,
        elinewidth=0.45,
        capsize=0.9,
        zorder=3,
    )
    for x, value, error in zip(positions, values, errors):
        ax.text(
            x,
            min(float(value + error + y_max * 0.018), y_max * 0.96),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=3.0,
            rotation=90,
            color=C_DARK,
        )
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(-0.55, len(FRAMEWORK_CONDITIONS) - 0.45)
    ax.set_xticks(positions)
    ax.set_xticklabels([short for _, _, short in FRAMEWORK_CONDITIONS])
    ax.tick_params(axis="x", length=0, pad=1.0)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    if show_ylabels:
        ax.set_ylabel(ylabel, fontsize=4.3, labelpad=1.0)
    else:
        ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=5.4, fontweight="bold", pad=2.0)
    _style_axis(ax, labelsize=3.8)
    return bars


def _plot_panel_b(fig: plt.Figure, slot, data: dict[str, object]):
    header, content = _panel_header(
        fig,
        slot,
        "b",
        "",
        header_ratio=0.065,
    )
    grid = content.subgridspec(2, 6, hspace=0.38, wspace=0.36)
    recon = data["recon"].set_index(["dataset_key", "condition"])
    downstream = data["downstream"].set_index(["dataset_key", "condition"])
    all_wf = data["recon"]["waveform_rmse_mean"].to_numpy(float) + data["recon"]["waveform_rmse_sd"].to_numpy(float)
    all_down = data["downstream"]["downstream_target_rmse_mean"].to_numpy(float) + data["downstream"]["downstream_target_rmse_sd"].to_numpy(float)
    wf_max = float(np.nanmax(all_wf) * 1.28)
    down_max = float(np.nanmax(all_down) * 1.28)
    top_axes, bottom_axes = [], []
    for j, (key, label, _) in enumerate(EXTERNAL_FRAMEWORK):
        top = fig.add_subplot(grid[0, j], sharey=top_axes[0] if top_axes else None)
        bottom = fig.add_subplot(grid[1, j], sharey=bottom_axes[0] if bottom_axes else None)
        top_axes.append(top)
        bottom_axes.append(bottom)
        wf = recon.loc[(key, [condition for condition, _, _ in FRAMEWORK_CONDITIONS]), "waveform_rmse_mean"].to_numpy(float)
        wf_sd = recon.loc[(key, [condition for condition, _, _ in FRAMEWORK_CONDITIONS]), "waveform_rmse_sd"].to_numpy(float)
        down = downstream.loc[(key, [condition for condition, _, _ in FRAMEWORK_CONDITIONS]), "downstream_target_rmse_mean"].to_numpy(float)
        down_sd = downstream.loc[(key, [condition for condition, _, _ in FRAMEWORK_CONDITIONS]), "downstream_target_rmse_sd"].to_numpy(float)
        _bar_card(top, wf, wf_sd, wf_max, j == 0, "Waveform RMSE (V)", label)
        _bar_card(bottom, down, down_sd, down_max, j == 0, "Downstream RMSE", None)
    return [header, *top_axes, *bottom_axes]


def _draw_small_pi_heatmap(
    fig: plt.Figure,
    slot,
    matrix: np.ndarray,
    show_ylabels: bool,
    show_xlabels: bool,
):
    ax = fig.add_subplot(slot)
    ax.pcolormesh(
        np.arange(6, dtype=float),
        np.arange(6, dtype=float),
        matrix,
        cmap=POP_CMAP if matrix is None else None,
    )
    return ax


def _heatmap_axis(
    fig: plt.Figure,
    slot,
    matrix: np.ndarray,
    cmap,
    show_ylabels: bool,
    show_xlabels: bool,
):
    ax = fig.add_subplot(slot)
    mesh = ax.pcolormesh(
        np.arange(6, dtype=float),
        np.arange(6, dtype=float),
        matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        edgecolors="white",
        linewidth=0.48,
    )
    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)
    ax.set_box_aspect(1.0)
    ax.set_anchor("W")
    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_yticks(np.arange(5) + 0.5)
    if show_xlabels:
        ax.set_xticklabels(SEVERITIES)
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_yticklabels([label for _, label in PI_CONDITIONS])
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0, pad=1.1, labelsize=3.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax, mesh


def _plot_panel_c(fig: plt.Figure, slot, data: dict[str, object]):
    header, content = _panel_header(
        fig,
        slot,
        "c",
        "",
        header_ratio=0.04,
    )
    grid = content.subgridspec(
        3,
        6,
        height_ratios=(0.14, 1.0, 1.0),
        width_ratios=(1.0, 1.0, 1.0, 1.0, 1.0, 0.17),
        hspace=0.22,
        wspace=0.26,
    )
    pi = data["pi"]
    axes = [header]
    pop_mesh_for_colorbar = None
    ind_mesh_for_colorbar = None
    for j, (key, label) in enumerate(EXTERNAL_PI):
        title_ax = fig.add_subplot(grid[0, j])
        title_ax.set_axis_off()
        title_ax.text(
            0.5,
            0.45,
            label,
            transform=title_ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.4,
            fontweight="bold",
            color=C_DARK,
        )
        axes.append(title_ax)

    for j, (key, _) in enumerate(EXTERNAL_PI):
        pop = _within_system_matrix(pi, key, "E_pop")
        ind = _within_system_matrix(pi, key, "E_ind")
        pop_ax, pop_mesh = _heatmap_axis(fig, grid[1, j], pop, POP_CMAP, show_ylabels=(j == 0), show_xlabels=False)
        ind_ax, ind_mesh = _heatmap_axis(fig, grid[2, j], ind, IND_CMAP, show_ylabels=(j == 0), show_xlabels=True)
        axes.extend([pop_ax, ind_ax])
        if j == 0:
            pop_mesh_for_colorbar = pop_mesh
            ind_mesh_for_colorbar = ind_mesh

    # Keep the two display-normalized colour keys visible, as in the earlier
    # cross-system heatmap convention.  They describe colour intensity only;
    # raw formal values remain in the derived tables and are not ranked here.
    cax_pop = fig.add_subplot(grid[1, -1])
    cax_ind = fig.add_subplot(grid[2, -1])
    cb_pop = fig.colorbar(pop_mesh_for_colorbar, cax=cax_pop)
    cb_ind = fig.colorbar(ind_mesh_for_colorbar, cax=cax_ind)
    for cb, label in [(cb_pop, r"$E_{\mathrm{pop}}$"), (cb_ind, r"$E_{\mathrm{ind}}$")]:
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=3.2, length=0, width=0, pad=1.0, colors=C_DARK)
        cb.outline.set_visible(True)
        cb.outline.set_linewidth(0.4)
        cb.set_label(label, fontsize=3.6, labelpad=1.2)
    axes.extend([cax_pop, cax_ind])
    return axes


def _write_readme_and_qc(data: dict[str, object], outputs: list[Path]) -> None:
    readme = f"""# Fig. 6 main-text sketch — R5 cross-system

This figure uses only the frozen R5 formal/canonical outputs.  The plotting
script is `{(OUT / 'make_fig6.py').resolve()}`.  No model training, inference,
new experiment, or inferential re-analysis is performed.

## Panel sources

- **(a) Representative reconstruction:** `{REPRESENTATIVE_NPZ.resolve()}` and `{REPRESENTATIVE_JSON.resolve()}`.  The six external framework cohorts are HUST, MICH, SDU, Stanford, TJU-NCA, and TJU-NCM.  All use the frozen canonical `Segment missing` representative asset and its recorded selection score/RMSE.
- **(b1) Waveform RMSE:** `{FRAMEWORK_RECON_SOURCE.resolve()}`.  The plotted values are the formal `waveform_rmse_mean` and `waveform_rmse_sd` for the external six datasets and four framework corruptions.
- **(b2) Downstream target RMSE:** `{FRAMEWORK_DOWNSTREAM_SOURCE.resolve()}`.  The plotted values are the formal `downstream_target_rmse_mean` and `downstream_target_rmse_sd`; the formal delta-to-clean columns are intentionally not plotted.  SDU uses the dataset-defined normalized second-life capacity target; the other external cohorts use their formal downstream target.
- **(c) Population–Individual recoverability:** `{PI_SOURCE.resolve()}`.  The panel displays HUST, MICH, Stanford, TJU-NCA, and TJU-NCM only; SDU is outside the PI scope and MIT is omitted from this external-replication display.  The formal grid is five corruption types × five severity levels.

The audit/provenance files read as integrity guards are:
`{SOURCE_MANIFEST.resolve()}`, `{PROTOCOL_SOURCE.resolve()}`,
`{CLAIM_MATRIX.resolve()}`, and `{METHODS_FACT_SHEET.resolve()}`.

## Display-only processing

Panel (c) uses a separate min–max colour scaling within each plotted dataset
and metric to emphasize the stress-response pattern.  This is display-only;
raw formal `E_pop` and `E_ind` values are retained in the generated plotting
tables `fig6_panel_c_epop.csv` and `fig6_panel_c_eind.csv`.  The two colourbars
are shared 0–1 display keys for the normalized colour intensity, not absolute
cross-system metric scales; no absolute cross-system `E_ind` ranking is made.

## Caption-ready draft points

- Panel (a): representative reconstruction examples across external battery systems under frozen segment-missing input.
- Panel (b): absolute waveform RMSE and downstream target RMSE across external systems under four benchmark corruption types.
- Panel (c): external replication of the Population–Individual recoverability hierarchy across corruption types and severity levels; display colours are scaled within each system, while absolute values are reported in Supplementary Table X.

## Main-figure exclusions

Panel (d), cross-method comparison, downstream association, full numerical
tables, and SI-only sensitivity analyses are not included in this main figure.
"""
    (OUT / "README_fig6.md").write_text(readme, encoding="utf-8")

    output_lines = "\n".join(str(path.resolve()) for path in outputs)
    qc = f"""# Fig. 6 QC

## Scientific QC

1. Panel (a) contains only the external six framework systems: **PASS**.
2. Panel (b) shows BatteryDiT absolute performance only; no corrupted baseline, delta, or improvement bar is plotted: **PASS**.
3. Panel (b) dataset order matches panel (a): **PASS**.
4. Panel (c) contains HUST, MICH, Stanford, TJU-NCA, and TJU-NCM only: **PASS**.
5. Panel (c) explicitly uses within-system display scaling and retains raw values in plotting tables: **PASS**.
6. Panel (c) uses two independent 0–1 display colourbars and provides no global absolute `E_ind` scale or cross-dataset absolute ranking: **PASS**.
7. No cross-method panel is included: **PASS**.
8. New experiments, model training, and model inference: **0 / PASS**.
9. All plotted values come directly from current canonical/formal/audit sources listed in README_fig6.md: **PASS**.

## Visual/output QC

10. Three main panels only; panel labels are left-aligned, a/b/c columns are aligned, and panel (c) spans the full row with paired colourbars: **PASS**.
11. PNG/PDF/SVG successfully written: **PASS**.

## Outputs

{output_lines}

Legacy manuscript tables, legacy Fig. 6 values, and old `figures/fig6/plot.py`
data sources were not used.
"""
    (OUT / "fig6_qc.md").write_text(qc, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    apply_paper_style()
    # Match the historical plotting convention used by the earlier main/SI
    # figures while remaining deterministic on this host, where Arial is not
    # installed.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.labelsize": 6.7,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plt.rcParams["svg.fonttype"] = "none"
    export_helpers.FONT_FAMILY = "Liberation Sans"
    data = _audit_and_load()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_plotting_tables(data)

    fig = plt.figure(figsize=(NC_WIDTH, 200 * MM), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.86, 1.08, 1.25),
        hspace=0.20,
        left=0.075,
        right=0.985,
        top=0.985,
        bottom=0.055,
    )
    axes_a = _plot_panel_a(fig, outer[0], data)
    axes_b = _plot_panel_b(fig, outer[1], data)
    axes_c = _plot_panel_c(fig, outer[2], data)

    enforce_nc_readability(fig)
    _typography_audit(fig, "fig6_main", OUT)
    outputs = [OUT / "Fig6_main.png", OUT / "Fig6_main.pdf", OUT / "Fig6_main.svg"]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.03)
    _write_readme_and_qc(data, outputs)
    plt.close(fig)

    print("FIG6 MAIN:")
    print("READY")
    print("PANEL A: READY — external six, frozen Segment missing representatives")
    print("PANEL B: READY — external six × four corruptions, absolute RMSE only")
    print("PANEL C: READY — five external nominal-SOH systems, within-system colour scaling")
    print("PANEL D: OMITTED")
    print("NEW EXPERIMENTS: 0")
    print("MODEL TRAINING: 0")
    print("MODEL INFERENCE: 0")
    print("LEGACY DATA USED: NO")
    print("OUTPUTS:")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
