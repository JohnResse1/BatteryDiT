"""Reproduce the PC-mask physics-consistency routing schematic.

This figure is a deterministic methods illustration, not an empirical result.
All signals and masks are generated locally with the fixed seed below; the
module has no dependency on the original repository, model checkpoints, or
external data files.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

from figures.common import export as export_helpers


FIGURE_ID = "figS10"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / FIGURE_ID
OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MM = 1.0 / 25.4
CD = 183 * MM
RNG_SEED = 42
SIGNAL_LENGTH = 1000

# Copied from the original SI schematic's local plotting implementation.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 6.2,
    "axes.labelsize": 6.3,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 5.7,
    "ytick.labelsize": 5.7,
    "legend.fontsize": 5.2,
    "axes.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "axes.grid": True,
    "grid.alpha": 0.11,
    "grid.linewidth": 0.35,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C_BLUE = "#2166AC"
C_BLUE_LIGHT = "#8BBFDB"
C_RED = "#D85C48"
C_RED_LIGHT = "#EFB097"
C_NEUTRAL = "#7F878D"
C_REFERENCE = "#111111"
C_GRID = "#B6BEC4"


def style_axis(axis: plt.Axes, grid_axis: str = "both") -> None:
    axis.tick_params(pad=1)
    axis.grid(True, axis=grid_axis, alpha=0.08, linewidth=0.3)
    axis.set_axisbelow(True)


def add_panel_label(axis: plt.Axes, label: str, x: float = -0.075, y: float = 1.04):
    return axis.text(
        x, y, label, transform=axis.transAxes,
        fontsize=8, fontweight="bold", va="bottom", ha="left",
    )


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open index intervals for true runs in a Boolean mask."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def build_signals() -> dict[str, np.ndarray | float | int]:
    """Build the deterministic schematic signals and routing quantities."""
    rng = np.random.default_rng(RNG_SEED)
    length = SIGNAL_LENGTH
    t = np.linspace(0, 1, length)

    v_clean = (
        2.0
        + 0.8 * t
        + 0.5 * np.tanh((t - 0.6) * 8)
        + 0.2 * (1 - np.exp(-t / 0.1))
        + 0.003 * np.sin(t * 30)
    )
    v_phys = 2.0 + 0.75 * t + 0.48 * np.tanh((t - 0.6) * 8)
    v_observed = v_clean.copy()

    noise_region = (t >= 0.02) & (t < 0.32)
    gap_region = (t >= 0.56) & (t < 0.74)
    drift_region = (t >= 0.82) & (t <= 1.0)
    spike_indices = np.asarray([378, 422])

    v_observed[noise_region] += rng.normal(0.0, 0.025, noise_region.sum())
    v_observed[spike_indices] += np.asarray([0.38, -0.30])
    v_observed[gap_region] = 0.0
    v_observed[drift_region] += rng.normal(0.0, 0.002, drift_region.sum()).cumsum()

    metadata_mask = np.ones(length)
    metadata_mask[gap_region] = 0.0
    is_valid = (np.abs(v_observed) > 1e-6) & np.isfinite(v_observed)

    v_difference = np.abs(np.diff(v_observed, prepend=v_observed[0]))
    v_scale = np.clip(np.mean(np.abs(v_observed[is_valid])), 2.0, 5.0)
    discrepancy = np.maximum(v_difference / v_scale - 0.01, 0.0)

    # Make schematic events legible at journal scale, as in the original.
    discrepancy[378] = 0.18
    discrepancy[379] = 0.09
    discrepancy[422] = 0.14
    discrepancy[423] = 0.07
    gap_start = np.flatnonzero(gap_region)[0]
    gap_stop = np.flatnonzero(gap_region)[-1] + 1
    discrepancy[gap_start] = 0.12
    discrepancy[min(gap_stop, length - 1)] = 0.10

    sigma_po = 1.0
    sigma_sensor_sq = 0.01 + np.expm1(25.0 * discrepancy)
    kappa = sigma_po / (sigma_po + sigma_sensor_sq)
    soft_confidence = np.where(
        metadata_mask > 0.5,
        np.clip(kappa, 0.5, 1.0) * is_valid,
        0.047,
    )
    hard_mask = (soft_confidence > 0.5).astype(float)

    v_interpolated = v_observed.copy()
    v_interpolated[gap_region] = np.interp(
        t[gap_region], t[~gap_region], v_observed[~gap_region]
    )
    v_base = hard_mask * v_interpolated + (1 - hard_mask) * v_phys

    return {
        "t": t,
        "v_clean": v_clean,
        "v_phys": v_phys,
        "v_observed": v_observed,
        "gap_region": gap_region,
        "spike_indices": spike_indices,
        "discrepancy": discrepancy,
        "hard_mask": hard_mask,
        "v_base": v_base,
        "delta_crit": float(np.log(1.99) / 25.0),
        "sigma_po": sigma_po,
    }


def _draw_figure(signals: dict[str, np.ndarray | float | int]) -> plt.Figure:
    t = np.asarray(signals["t"])
    v_clean = np.asarray(signals["v_clean"])
    v_phys = np.asarray(signals["v_phys"])
    v_observed = np.asarray(signals["v_observed"])
    gap_region = np.asarray(signals["gap_region"], dtype=bool)
    spike_indices = np.asarray(signals["spike_indices"], dtype=int)
    discrepancy = np.asarray(signals["discrepancy"])
    hard_mask = np.asarray(signals["hard_mask"])
    v_base = np.asarray(signals["v_base"])
    delta_crit = float(signals["delta_crit"])
    sigma_po = float(signals["sigma_po"])
    length = len(t)

    fig = plt.figure(figsize=(CD, 118 * MM), facecolor="white")
    grid = GridSpec(
        3, 2, figure=fig,
        width_ratios=[3.08, 1.08],
        height_ratios=[1.32, 0.74, 0.74],
        left=0.072, right=0.982, bottom=0.095, top=0.945,
        hspace=0.38, wspace=0.25,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(grid[2, 0], sharex=ax_a)
    ax_d = fig.add_subplot(grid[:, 1])

    region_specs = [
        (0.02, 0.32, C_BLUE_LIGHT, "Measurement noise"),
        (0.56, 0.74, C_RED_LIGHT, "Missing segment"),
        (0.82, 1.00, "#BCB7B7", "Slow drift"),
    ]
    for axis in (ax_a, ax_b, ax_c):
        for left, right, color, _label in region_specs:
            axis.axvspan(left, right, color=color, alpha=0.10, lw=0, zorder=0)

    # (a) Candidate base signal
    observed_display = v_observed.copy()
    observed_display[gap_region] = np.nan
    ax_a.plot(t, observed_display, color=C_NEUTRAL, lw=0.58, alpha=0.78,
              label="Observed voltage", zorder=2)
    ax_a.plot(t, v_phys, color=C_RED, lw=0.82, ls="--", alpha=0.82,
              label=r"Physics prior $V_{\mathrm{phys}}(I)$", zorder=3)
    ax_a.plot(t, v_base, color=C_BLUE, lw=1.18,
              label=r"Routed base $x_{\mathrm{base}}$", zorder=4)
    ax_a.plot(t, v_clean, color=C_REFERENCE, lw=0.72, alpha=0.92,
              label="Clean reference", zorder=5)

    for left, right, _color, label in region_specs:
        if label == "Slow drift":
            continue
        label_color = {
            "Measurement noise": C_BLUE,
            "Missing segment": C_RED,
        }[label]
        ax_a.text(
            (left + right) / 2, 0.965, label,
            transform=ax_a.get_xaxis_transform(), ha="center", va="top",
            color=label_color, fontsize=5.3, fontweight="bold",
        )

    for index in spike_indices:
        ax_a.annotate(
            "", xy=(t[index], v_observed[index]),
            xytext=(t[index], v_observed[index] + 0.20),
            arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=0.55, mutation_scale=5),
            zorder=8,
        )

    inset = ax_a.inset_axes([0.315, 0.535, 0.235, 0.365])
    zoom = slice(360, 440)
    inset.plot(t[zoom], observed_display[zoom], color=C_NEUTRAL, lw=0.55, alpha=0.76)
    inset.plot(t[zoom], v_base[zoom], color=C_BLUE, lw=1.05)
    inset.plot(t[zoom], v_clean[zoom], color=C_REFERENCE, lw=0.65)
    inset.scatter(
        t[spike_indices], v_observed[spike_indices], marker="x", s=13,
        linewidths=0.85, color=C_RED, zorder=7, clip_on=False,
    )
    inset.set_xlim(t[360], t[439])
    inset.set_ylim(1.89, 2.55)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)
    inset.tick_params(length=0)
    for spine in inset.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.5)
    inset.text(
        0.50, 0.94, "Unsupported spikes rejected", transform=inset.transAxes,
        ha="center", va="top", fontsize=4.55, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.25),
    )

    ax_a.annotate(
        "prior fill", xy=(0.65, np.interp(0.65, t, v_base)),
        xytext=(0.65, 3.50), ha="center", va="bottom", color=C_RED,
        fontsize=5.15,
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.5, mutation_scale=5),
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(1.86, 3.86)
    ax_a.set_ylabel("Voltage (V)", labelpad=2)
    ax_a.tick_params(axis="x", labelbottom=False)
    style_axis(ax_a)
    add_panel_label(ax_a, "(a)", x=-0.062, y=1.035)

    legend_handles = [
        Line2D([0], [0], color=C_REFERENCE, lw=0.8, label="Clean reference"),
        Line2D([0], [0], color=C_NEUTRAL, lw=0.7, label="Observed voltage"),
        Line2D([0], [0], color=C_RED, lw=0.9, ls="--",
               label=r"Physics prior $V_{\mathrm{phys}}(I)$"),
        Line2D([0], [0], color=C_BLUE, lw=1.2,
               label=r"Routed base $x_{\mathrm{base}}$"),
    ]
    ax_a.legend(
        handles=legend_handles, loc="lower right", ncol=1,
        handlelength=1.8, columnspacing=0.9, handletextpad=0.45,
        labelspacing=0.25, frameon=True, facecolor="white",
        edgecolor="none", framealpha=0.82, borderpad=0.25,
    )

    # (b) Fixed binary route
    ax_b.fill_between(t, 0, hard_mask, step="mid", color=C_BLUE, alpha=0.22, lw=0)
    ax_b.fill_between(t, hard_mask, 1, step="mid", color=C_RED, alpha=0.14, lw=0)
    ax_b.step(t, hard_mask, where="mid", color=C_BLUE, lw=0.92, zorder=4)
    for start, stop in contiguous_regions(hard_mask < 0.5):
        is_gap = (stop - start) > 50
        ax_b.axvspan(
            t[start], t[min(stop, length - 1)],
            color=C_RED, alpha=0.22 if is_gap else 0.48, lw=0, zorder=3,
        )
    ax_b.axhline(0.5, color=C_GRID, lw=0.45, ls=":")
    ax_b.text(0.16, 0.76, r"$m=1$", transform=ax_b.transAxes,
              color=C_BLUE, fontsize=5.15, fontweight="bold")
    ax_b.text(0.63, 0.13, r"$m=0$", transform=ax_b.transAxes,
              color=C_RED, fontsize=5.15, fontweight="bold")
    ax_b.set_ylim(-0.06, 1.08)
    ax_b.set_yticks([0, 0.5, 1])
    ax_b.set_ylabel("Hard mask", labelpad=2)
    ax_b.tick_params(axis="x", labelbottom=False)
    style_axis(ax_b)
    add_panel_label(ax_b, "(b)", x=-0.062, y=1.035)

    # (c) Local discrepancy
    ax_c.plot(t, discrepancy, color=C_RED, lw=0.78, alpha=0.92)
    ax_c.axhline(delta_crit, color=C_REFERENCE, lw=0.58, ls="--")
    ax_c.scatter(t[spike_indices], discrepancy[spike_indices], marker="x", s=14,
                 linewidths=0.85, color=C_RED, zorder=6)
    ax_c.text(0.455, delta_crit + 0.008, r"$\delta_{\mathrm{crit}}=0.0275$",
              fontsize=4.85, color="#555555", ha="left", va="bottom")
    ax_c.set_ylim(-0.005, 0.205)
    ax_c.set_ylabel(r"Discrepancy $\delta_t$", labelpad=2)
    ax_c.set_xlabel("Normalized charging time", labelpad=2)
    style_axis(ax_c)
    add_panel_label(ax_c, "(c)", x=-0.062, y=1.035)

    # (d) Reliability score and the hard threshold
    delta = np.linspace(0, 0.16, 500)
    sensor_variance = 0.01 + np.expm1(25.0 * delta)
    kappa_curve = sigma_po / (sigma_po + sensor_variance)
    ax_d.axvspan(0, delta_crit, color=C_BLUE, alpha=0.10, lw=0)
    ax_d.axvspan(delta_crit, 0.16, color=C_RED, alpha=0.10, lw=0)
    sensor_region = delta <= delta_crit
    prior_region = delta >= delta_crit
    ax_d.plot(delta[sensor_region], kappa_curve[sensor_region],
              color=C_BLUE, lw=1.42, zorder=4)
    ax_d.plot(delta[prior_region], kappa_curve[prior_region],
              color=C_RED, lw=1.42, zorder=4)
    ax_d.axhline(0.5, color=C_REFERENCE, lw=0.62, ls="--", zorder=2)
    ax_d.axvline(delta_crit, color=C_GRID, lw=0.55, ls=":", zorder=2)
    ax_d.scatter([delta_crit], [0.5], s=13, color=C_REFERENCE, zorder=5)
    ax_d.annotate(r"$\delta_{\mathrm{crit}}$", xy=(delta_crit, 0.5),
                  xytext=(0.057, 0.635), ha="center", va="bottom", fontsize=5.0,
                  arrowprops=dict(arrowstyle="->", color=C_NEUTRAL, lw=0.5,
                                  mutation_scale=5))
    ax_d.legend(
        handles=[
            Line2D([], [], color=C_BLUE, lw=1.2, label=r"$m=1$: sensor route"),
            Line2D([], [], color=C_RED, lw=1.0, label=r"$m=0$: prior route"),
        ],
        loc="upper right", frameon=False, handlelength=1.6,
        handletextpad=0.35, labelspacing=0.25, borderaxespad=0.35,
        fontsize=5.0,
    )
    ax_d.set_xlim(0, 0.16)
    ax_d.set_ylim(0, 1.02)
    ax_d.set_xlabel(r"Discrepancy $\delta$", labelpad=2)
    ax_d.set_ylabel(r"Reliability score $\kappa(\delta)$", labelpad=2)
    style_axis(ax_d)
    add_panel_label(ax_d, "(d)", x=-0.18, y=1.012)

    return fig


def _write_local_manifest(signals: dict[str, np.ndarray | float | int]) -> None:
    manifest = {
        "figure": FIGURE_ID,
        "figure_type": "deterministic methods schematic",
        "source_script": "figures/figS10/plot.py",
        "external_data": False,
        "model_training": 0,
        "model_inference": 0,
        "rng_seed": RNG_SEED,
        "signal_length": SIGNAL_LENGTH,
        "noise_region": [0.02, 0.32],
        "missing_segment": [0.56, 0.74],
        "slow_drift_region": [0.82, 1.0],
        "spike_indices": [int(v) for v in np.asarray(signals["spike_indices"])],
        "delta_crit": float(signals["delta_crit"]),
        "notes": "All signal, discrepancy, confidence, and hard-mask arrays are generated locally by the script.",
    }
    (OUT / "figS10_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def draw() -> dict[str, str]:
    signals = build_signals()
    _write_local_manifest(signals)
    figure = _draw_figure(signals)
    previous_font_family = export_helpers.FONT_FAMILY
    export_helpers.FONT_FAMILY = "Liberation Sans"
    try:
        outputs = export_helpers.export_figure(figure, FIGURE_ID)
    finally:
        export_helpers.FONT_FAMILY = previous_font_family
    plt.close(figure)
    return outputs


if __name__ == "__main__":
    print(json.dumps(draw(), indent=2))
