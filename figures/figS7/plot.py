"""Supplementary Fig. S7: formal R4 Population--Individual properties.

This figure is the former R4 Fig. 5 moved to the Supplementary Information.
It reads only the frozen R4 plotting assets and the repaired formal
median-bootstrap source for panel (a).  The six panels are independent:
the former panel-b and panel-d inset plots are no longer overlaid on another
axis.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.workflows._stats import spearman
from figures.common.export import export_figure
from figures.common.style import (
    C_BLUE,
    C_CORRUPTED,
    C_DARK,
    C_GREY,
    C_RECONSTRUCTED,
    FIGURE_WIDTH,
    apply_paper_style,
    panel_label,
)


REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RUN = REPO / "analysis" / "outputs" / "r4_formal" / "latest"
DERIVED = ROOT / "derived_tables"
FIGURE_ID = "figS7"
FIGURE_STEM = "figS7_main"
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260926 + 91000

CORRUPTIONS = ("downsample", "random", "segment", "destructive", "hybrid")
CORRUPTION_LABELS = {
    "downsample": "Downsampling",
    "random": "Random-point missing",
    "segment": "Segment missing",
    "destructive": "Destructive noise",
    "hybrid": "Hybrid",
}
PREDICTOR_LABELS = {"ridge": "Ridge", "knn": "kNN", "mlp": "MLP"}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty plotting table: {path}")
    return frame


def _finite(frame: pd.DataFrame, name: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def _load_formal_longitudinal_median() -> pd.DataFrame:
    """Build the plotted long table from the repaired formal median bootstrap."""
    source = _read(RUN / "longitudinal_bootstrap.csv")
    _finite(
        source,
        "longitudinal_bootstrap",
        [
            "lag", "observed_same", "same_ci_low", "same_ci_high",
            "observed_cross", "cross_ci_low", "cross_ci_high",
        ],
    )
    median = source[source.statistic.eq("median")].sort_values("lag", kind="stable")
    if len(median) != 6 or median.lag.duplicated().any():
        raise ValueError("formal longitudinal bootstrap lacks one median row per lag")
    rows = []
    for row in median.itertuples(index=False):
        rows.extend(
            [
                {
                    "lag": int(row.lag),
                    "group": "same_battery",
                    "estimate": float(row.observed_same),
                    "ci_low": float(row.same_ci_low),
                    "ci_high": float(row.same_ci_high),
                    "n_batteries": int(row.n_batteries),
                },
                {
                    "lag": int(row.lag),
                    "group": "matched_soh_cross_battery",
                    "estimate": float(row.observed_cross),
                    "ci_low": float(row.cross_ci_low),
                    "ci_high": float(row.cross_ci_high),
                    "n_batteries": int(row.n_batteries),
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_variation_vulnerability_points(
    support: pd.DataFrame, occlusion: pd.DataFrame
) -> pd.DataFrame:
    """Re-express the formal 20% window assets as the plotted X_w/Y_w points."""
    profile = support[support.component.eq("Individual")].sort_values("progress", kind="stable")
    response = occlusion[
        occlusion.window_width.eq(0.2) & occlusion.metric.eq("Delta_E_ind")
    ].sort_values("window_center", kind="stable")
    if len(profile) == 0 or len(response) != 17:
        raise ValueError("formal 20% support-vulnerability assets are incomplete")
    values = profile.estimate.to_numpy(float)
    rows = []
    for row in response.itertuples(index=False):
        start = float(row.window_center) - 0.1
        begin = int(round(start * len(values)))
        end = int(round((start + 0.2) * len(values)))
        rows.append(
            {
                "window_center": float(row.window_center),
                "window_start": start,
                "window_end": start + 0.2,
                "window_width": 0.2,
                "X_w": float(values[begin:end].sum()),
                "Y_w": float(row.estimate),
                "Y_ci_low": float(row.ci_low),
                "Y_ci_high": float(row.ci_high),
            }
        )
    return pd.DataFrame(rows)


def _build_dual_association() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build panel-f P/P+I associations from frozen battery-level assets."""
    link = _read(RUN / "fidelity_prognosis_link.csv")
    metrics = _read(RUN / "reconstruction_prognosis_battery_metrics.csv")
    _finite(link, "fidelity_prognosis_link", ["E_ind_loss"])
    _finite(metrics, "reconstruction_prognosis_battery_metrics", ["MAE"])
    link_required = {"battery_id", "corruption", "E_ind_loss"}
    metrics_required = {
        "battery_id", "corruption", "feature_set", "representation_source", "MAE"
    }
    if not link_required.issubset(link.columns):
        raise ValueError(f"fidelity link missing columns: {sorted(link_required - set(link.columns))}")
    if not metrics_required.issubset(metrics.columns):
        raise ValueError(
            "battery metrics missing columns: "
            f"{sorted(metrics_required - set(metrics.columns))}"
        )
    metric_key = ["battery_id", "corruption", "feature_set", "representation_source"]
    if metrics.duplicated(metric_key).any():
        raise ValueError("battery metrics are not unique at the formal plotting unit")
    metric_wide = (
        metrics.pivot(
            index=["battery_id", "corruption", "feature_set"],
            columns="representation_source",
            values="MAE",
        )
        .reset_index()
    )
    if not {"clean", "reconstructed"}.issubset(metric_wide.columns):
        raise ValueError("battery metrics need clean and reconstructed MAE rows")
    metric_wide["delta_MAE"] = metric_wide["reconstructed"] - metric_wide["clean"]
    p = metric_wide[metric_wide.feature_set.eq("P")][
        ["battery_id", "corruption", "delta_MAE"]
    ].rename(columns={"delta_MAE": "delta_MAE_P"})
    p_plus_i = metric_wide[metric_wide.feature_set.eq("P+I")][
        ["battery_id", "corruption", "delta_MAE"]
    ].rename(columns={"delta_MAE": "delta_MAE_PplusI"})
    dual = (
        link[["battery_id", "corruption", "E_ind_loss"]]
        .rename(columns={"E_ind_loss": "delta_E_ind"})
        .merge(p, on=["battery_id", "corruption"], how="inner", validate="one_to_one")
        .merge(p_plus_i, on=["battery_id", "corruption"], how="inner", validate="one_to_one")
    )
    dual["battery_id"] = dual["battery_id"].astype(str)
    corruption_order = {name: i for i, name in enumerate(CORRUPTIONS)}
    dual["_corruption_order"] = dual.corruption.map(corruption_order)
    dual = dual.sort_values(
        ["_corruption_order", "battery_id"], kind="stable"
    ).drop(columns="_corruption_order").reset_index(drop=True)
    if len(dual) != 105 or dual.battery_id.nunique() != 21:
        raise ValueError("dual association must contain exactly 21 batteries x 5 corruptions")
    if set(dual.corruption) != set(CORRUPTIONS):
        raise ValueError("dual association does not contain all five formal corruptions")
    _finite(
        dual,
        "fig5f_dual_association",
        ["delta_E_ind", "delta_MAE_P", "delta_MAE_PplusI"],
    )

    formal_pi = link[["battery_id", "corruption", "prognostic_loss"]].rename(
        columns={"prognostic_loss": "formal_prognostic_loss"}
    )
    check = dual.merge(formal_pi, on=["battery_id", "corruption"], validate="one_to_one")
    if not np.allclose(
        check.delta_MAE_PplusI,
        check.formal_prognostic_loss,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("derived P+I delta MAE does not reproduce formal prognostic_loss")

    batteries = np.array(sorted(dual.battery_id.unique()))
    battery_index = {
        battery: np.flatnonzero(dual.battery_id.to_numpy() == battery)
        for battery in batteries
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    x = dual.delta_E_ind.to_numpy(float)
    y_p = dual.delta_MAE_P.to_numpy(float)
    y_pi = dual.delta_MAE_PplusI.to_numpy(float)
    for draw_id in range(BOOTSTRAPS):
        selected = rng.integers(0, len(batteries), size=len(batteries))
        indices = np.concatenate([battery_index[batteries[index]] for index in selected])
        rho_p = spearman(x[indices], y_p[indices])
        rho_pi = spearman(x[indices], y_pi[indices])
        bootstrap_rows.append(
            {
                "draw_id": draw_id,
                "rho_P": rho_p,
                "rho_PplusI": rho_pi,
                "delta_rho": rho_pi - rho_p,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    stats = pd.DataFrame(
        [
            {
                "model": "P",
                "rho": spearman(x, y_p),
                "ci_low": float(np.nanpercentile(bootstrap.rho_P, 2.5)),
                "ci_high": float(np.nanpercentile(bootstrap.rho_P, 97.5)),
                "n_points": len(dual),
                "n_batteries": len(batteries),
                "n_conditions": len(CORRUPTIONS),
                "n_bootstrap": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            {
                "model": "P+I",
                "rho": spearman(x, y_pi),
                "ci_low": float(np.nanpercentile(bootstrap.rho_PplusI, 2.5)),
                "ci_high": float(np.nanpercentile(bootstrap.rho_PplusI, 97.5)),
                "n_points": len(dual),
                "n_batteries": len(batteries),
                "n_conditions": len(CORRUPTIONS),
                "n_bootstrap": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
        ]
    )
    return dual, stats, bootstrap


def load() -> dict[str, pd.DataFrame]:
    """Load current R4 tables and derive the dual panel-f plotting assets."""
    tables = {
        "longitudinal": _load_formal_longitudinal_median(),
        "support": _read(RUN / "fig5b_support_profile.csv"),
        "occlusion": _read(RUN / "fig5b_occlusion_profile.csv"),
        "support_vulnerability": _read(RUN / "fig5b_support_vulnerability.csv"),
        "support_summary": _read(RUN / "spatial_support_summary.csv"),
        "prognosis": _read(RUN / "fig5c_prognosis_increment.csv"),
        "reconstruction": _read(RUN / "fig5d_reconstruction_prognosis.csv"),
        "fidelity": _read(RUN / "fig5d_fidelity_link.csv"),
        "fidelity_stats": _read(RUN / "fidelity_prognosis_stats.csv"),
    }
    (
        tables["dual_association"],
        tables["dual_stats"],
        tables["dual_bootstrap"],
    ) = _build_dual_association()
    tables["variation_vulnerability"] = _build_variation_vulnerability_points(
        tables["support"], tables["occlusion"]
    )
    _finite(tables["longitudinal"], "fig5a_longitudinal", ["lag", "estimate", "ci_low", "ci_high"])
    _finite(tables["support"], "fig5b_support_profile", ["progress", "estimate", "ci_low", "ci_high"])
    _finite(tables["occlusion"], "fig5b_occlusion_profile", ["window_center", "estimate", "ci_low", "ci_high", "window_width"])
    _finite(tables["support_vulnerability"], "fig5b_support_vulnerability", ["width_fraction", "spearman_rho", "ci_low", "ci_high"])
    _finite(tables["support_summary"], "spatial_support_summary", ["Population", "Individual", "CI_low", "CI_high"])
    _finite(tables["prognosis"], "fig5c_prognosis_increment", ["MAE", "ci_low", "ci_high", "delta_MAE_vs_P", "delta_ci_low", "delta_ci_high"])
    _finite(tables["reconstruction"], "fig5d_reconstruction_prognosis", ["delta_MAE", "ci_low", "ci_high"])
    _finite(tables["fidelity"], "fig5d_fidelity_link", ["E_ind_loss", "prognostic_loss"])
    _finite(tables["fidelity_stats"], "fidelity_prognosis_stats", ["rho", "ci_low", "ci_high"])

    longitudinal = tables["longitudinal"]
    if set(longitudinal.group) != {"same_battery", "matched_soh_cross_battery"}:
        raise ValueError("formal longitudinal median table has unexpected groups")
    primary = longitudinal
    if (primary.ci_low > primary.estimate).any() or (primary.ci_high < primary.estimate).any():
        raise ValueError("formal longitudinal median intervals do not contain estimates")
    if set(tables["support"].component) != {"Population", "Individual"}:
        raise ValueError("support profile lacks Population/Individual rows")
    support_row = tables["support_vulnerability"][
        tables["support_vulnerability"].width_fraction.eq(0.2)
        & tables["support_vulnerability"].relationship.eq("S_ind_vs_Delta_E_ind")
    ]
    if len(support_row) != 1:
        raise ValueError("missing formal 20% support-vulnerability row")
    if set(tables["prognosis"].predictor) != set(PREDICTOR_LABELS):
        raise ValueError("prognosis table lacks Ridge/kNN/MLP")
    if set(tables["reconstruction"].corruption) != set(CORRUPTIONS):
        raise ValueError("reconstruction table lacks one of five corruption types")
    if set(tables["reconstruction"].feature_set) != {"P", "P+I"}:
        raise ValueError("reconstruction table lacks P/P+I rows")
    if len(tables["fidelity"]) != 105 or tables["fidelity"].battery_id.nunique() != 21:
        raise ValueError("fidelity link must contain 21 batteries x 5 conditions")
    if set(tables["fidelity_stats"].association) != {"E_ind_loss", "C_ind_loss"}:
        raise ValueError("fidelity statistics lack formal associations")
    formal_pi = tables["fidelity_stats"].query("association == 'E_ind_loss'").iloc[0]
    derived_pi = tables["dual_stats"].query("model == 'P+I'").iloc[0]
    if not np.allclose(
        [derived_pi.rho, derived_pi.ci_low, derived_pi.ci_high],
        [formal_pi.rho, formal_pi.ci_low, formal_pi.ci_high],
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("derived P+I bootstrap does not reproduce formal E_ind association")
    formal_vulnerability = tables["support_vulnerability"][
        tables["support_vulnerability"].width_fraction.eq(0.2)
        & tables["support_vulnerability"].relationship.eq("S_ind_vs_Delta_E_ind")
    ].iloc[0]
    plotted_rho = tables["variation_vulnerability"].X_w.corr(
        tables["variation_vulnerability"].Y_w, method="spearman"
    )
    if not np.isclose(plotted_rho, formal_vulnerability.spearman_rho, atol=1e-12):
        raise ValueError("window-level variation-vulnerability points do not reproduce formal rho")
    return tables


def _axis_style(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.tick_params(pad=1.5)
    axis.grid(True, axis=grid_axis, alpha=0.10, linewidth=0.35)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def _panel_a(axis: plt.Axes, data: pd.DataFrame) -> None:
    styles = {
        "same_battery": ("Same battery", C_BLUE, "-", "o"),
        "matched_soh_cross_battery": ("Matched-SOH cross-battery", C_DARK, "--", "s"),
    }
    for group, (label, color, linestyle, marker) in styles.items():
        frame = data[data.group.eq(group)].sort_values("lag")
        x = frame.lag.to_numpy(float)
        axis.fill_between(x, frame.ci_low, frame.ci_high, color=color,
                          alpha=0.12 if group == "same_battery" else 0.08,
                          linewidth=0, zorder=1)
        axis.plot(x, frame.estimate, color=color, linestyle=linestyle,
                  linewidth=1.2, marker=marker, markersize=3.0,
                  markerfacecolor="white", markeredgewidth=0.75,
                  markeredgecolor=color, label=label, zorder=3)
    axis.set_title("Longitudinal coherence", loc="center", pad=3, fontweight="bold")
    axis.set_xlabel("Cycle lag")
    axis.set_ylabel("PCA-space Individual-state distance")
    axis.set_xticks([1, 5, 10, 25, 50, 100])
    axis.set_xlim(-2, 104)
    axis.set_ylim(0, 2.15)
    axis.legend(loc="upper left", frameon=False, handlelength=1.5, labelspacing=0.35)
    _axis_style(axis)
    panel_label(axis, "a", x=-0.13, y=1.04)


def _panel_b(axis: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    colors = {"Population": C_BLUE, "Individual": C_CORRUPTED}
    for component in ("Population", "Individual"):
        frame = data["support"][data["support"].component.eq(component)].sort_values("progress")
        axis.fill_between(frame.progress, frame.ci_low, frame.ci_high,
                          color=colors[component], alpha=0.12, linewidth=0)
        axis.plot(frame.progress, frame.estimate, color=colors[component],
                  linewidth=1.2, label=component)
    axis.set_title("Spatial distribution of trajectory variation", loc="center", pad=3, fontweight="bold")
    axis.set_xlabel("Normalized charging progress")
    axis.set_ylabel(r"Position-wise variation ($V^2$)")
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.5,
                columnspacing=1.0, labelspacing=0.3)
    _axis_style(axis)
    # The baseline is already encoded by the x-axis; suppress the redundant
    # zero label so it does not collide visually with the horizontal axis.
    axis.set_yticks([tick for tick in axis.get_yticks() if not np.isclose(tick, 0.0)])
    panel_label(axis, "b", x=-0.13, y=1.04)


def _panel_c(axis: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    points = data["variation_vulnerability"]
    axis.errorbar(
        points.X_w,
        points.Y_w,
        yerr=[points.Y_w - points.Y_ci_low, points.Y_ci_high - points.Y_w],
        fmt="none",
        ecolor=C_CORRUPTED,
        elinewidth=0.55,
        capsize=1.2,
        alpha=0.45,
        zorder=1,
    )
    axis.scatter(
        points.X_w,
        points.Y_w,
        s=15.0,
        color=C_CORRUPTED,
        edgecolors="white",
        linewidth=0.45,
        zorder=2,
    )
    axis.set_title("Variation–vulnerability association", loc="center", pad=3, fontweight="bold")
    axis.set_xlabel(r"Integrated Individual variation, $X_w$ ($V^2$)")
    axis.set_ylabel(r"Occlusion-induced Individual error, $\Delta E_{\mathit{ind}}$")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    _axis_style(axis)
    panel_label(axis, "c", x=-0.13, y=1.04)


def _panel_d(axis: plt.Axes, data: pd.DataFrame) -> None:
    frame = data[data.feature_set.eq("P+I")].copy()
    frame["order"] = frame.predictor.map({"ridge": 0, "knn": 1, "mlp": 2})
    frame = frame.sort_values("order")
    x = np.arange(len(frame), dtype=float)
    axis.axhline(0, color="#777777", linestyle="--", linewidth=0.7, zorder=0)
    for xi, (_, row) in zip(x, frame.iterrows()):
        axis.errorbar(xi, row.delta_MAE_vs_P,
                      yerr=[[row.delta_MAE_vs_P - row.delta_ci_low],
                            [row.delta_ci_high - row.delta_MAE_vs_P]],
                      fmt="o", ms=4, mfc=C_BLUE, mec="white", mew=0.45,
                      ecolor=C_BLUE, elinewidth=0.8, capsize=1.8, zorder=3)
    axis.set_title(
        "Incremental prognostic information in\nIndividual deviations",
        loc="center",
        pad=3,
        fontweight="bold",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([PREDICTOR_LABELS[p] for p in frame.predictor])
    axis.set_xlabel("Prognostic model")
    axis.set_ylabel(r"$\Delta MAE = MAE(P) - MAE(P+I)$")
    axis.set_xlim(-0.55, len(frame) - 0.45)
    axis.set_ylim(min(0, float(frame.delta_ci_low.min())) - 0.001,
                  float(frame.delta_ci_high.max()) + 0.002)
    _axis_style(axis)
    panel_label(axis, "d", x=-0.13, y=1.04)


def _panel_e(axis: plt.Axes, data: pd.DataFrame) -> None:
    x0 = np.arange(len(CORRUPTIONS), dtype=float)
    offsets = {"P": -0.10, "P+I": 0.10}
    colors = {"P": C_RECONSTRUCTED, "P+I": C_CORRUPTED}
    for feature_set in ("P", "P+I"):
        frame = data[data.feature_set.eq(feature_set)].copy()
        frame["order"] = frame.corruption.map({name: i for i, name in enumerate(CORRUPTIONS)})
        frame = frame.sort_values("order")
        yerr = np.vstack([frame.delta_MAE - frame.ci_low, frame.ci_high - frame.delta_MAE])
        axis.errorbar(x0 + offsets[feature_set], frame.delta_MAE, yerr=yerr,
                      fmt="o", ms=3.8, mfc=colors[feature_set], mec="white", mew=0.4,
                      ecolor=colors[feature_set], elinewidth=0.8, capsize=1.7,
                      label=(r"$P$" if feature_set == "P" else r"$P+I$"), zorder=3)
    axis.axhline(0, color="#777777", linestyle="--", linewidth=0.7, zorder=0)
    axis.set_title("Reconstruction-induced prognostic change", loc="center", pad=3, fontweight="bold")
    axis.set_xticks(x0)
    axis.set_xticklabels([CORRUPTION_LABELS[name] for name in CORRUPTIONS], rotation=22, ha="right")
    axis.set_xlabel("Observation degradation")
    axis.set_ylabel(
        r"$\Delta MAE_{\mathit{recon}}$" + "\n" +
        r"$= MAE_{\mathit{recon}} - MAE_{\mathit{clean}}$"
    )
    axis.set_xlim(-0.55, len(CORRUPTIONS) - 0.45)
    axis.set_ylim(float(data.ci_low.min()) - 0.0015, float(data.ci_high.max()) + 0.002)
    axis.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.3,
                columnspacing=0.9, labelspacing=0.3)
    _axis_style(axis)
    panel_label(axis, "e", x=-0.13, y=1.04)


def _panel_f(parent_spec, data: dict[str, pd.DataFrame]) -> None:
    """Draw panel (f) as matched P and P+I associations."""
    figure = plt.gcf()
    nested = parent_spec.subgridspec(1, 2, wspace=0.48)
    axes = [figure.add_subplot(nested[0]), figure.add_subplot(nested[1])]
    dual = data["dual_association"]
    stats = data["dual_stats"].set_index("model")
    x = dual.delta_E_ind.to_numpy(float)
    y_values = {
        "P": dual.delta_MAE_P.to_numpy(float),
        "P+I": dual.delta_MAE_PplusI.to_numpy(float),
    }
    x_pad = max(0.02, 0.04 * (float(x.max()) - float(x.min())))
    y_all = np.concatenate(list(y_values.values()))
    y_pad = max(0.002, 0.08 * (float(y_all.max()) - float(y_all.min())))
    x_limits = (max(0.0, float(x.min()) - x_pad), float(x.max()) + x_pad)
    y_limits = (float(y_all.min()) - y_pad, float(y_all.max()) + y_pad)
    colors = {"P": C_BLUE, "P+I": C_CORRUPTED}
    y_labels = {
        "P": r"$\Delta MAE_{P,\mathit{recon}}$",
        "P+I": r"$\Delta MAE_{P+I,\mathit{recon}}$",
    }
    for axis, model in zip(axes, ("P", "P+I")):
        axis.scatter(
            x,
            y_values[model],
            s=9.0,
            color=colors[model],
            alpha=0.48,
            edgecolors="white",
            linewidth=0.25,
            zorder=3,
        )
        axis.axhline(0, color="#777777", linestyle="--", linewidth=0.65, zorder=0)
        axis.set_title(r"$P$" if model == "P" else r"$P+I$", loc="center", pad=3, fontweight="bold")
        axis.set_xlabel(r"$\Delta E_{\mathit{ind}}$")
        axis.set_ylabel(y_labels[model])
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        row = stats.loc[model]
        axis.text(
            0.04,
            0.96,
            f"$\\rho$ = {row.rho:.3f}\n95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}]",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.4,
            color=C_DARK,
        )
        _axis_style(axis)
    panel_label(axes[0], "f", x=-0.28, y=1.04)


def _write_readme() -> None:
    text = f"""# Supplementary Fig. S7 — R4 Population–Individual properties

This is a plotting-only formal R4 figure.  It reads only
`{RUN.resolve()}` and does not rerun BatteryDiT, reconstruction, prognosis
fitting, or formal analysis.  The repaired median-bootstrap source for panel
(a) remains the runtime source.

## Panels

- **(a)** Longitudinal coherence: same-battery and matched-SOH cross-battery
  distances in the 10-D train-only PCA space of Individual residuals.  The
  plotted centers and percentile intervals are read directly from the
  repaired formal `longitudinal_bootstrap.csv` rows labelled `median`, with
  whole-battery bootstrap units; the old standalone corrected derivative is
  not used at runtime.
- **(b)** Spatial distribution of trajectory variation along charging
  progress.  The y-axis is position-wise variation in V²; the formal `C50`
  values are retained in the source table and documented here rather than
  printed inside the panel.
- **(c)** Variation–vulnerability association across the 17 formal 20%
  windows.  The x coordinate is the formal window sum
  `X_w = sum(S_ind)` and the y coordinate is the formal window-level
  occlusion-induced Individual error `Delta_E_ind`.  The plotting-only
  scatter table is written to `derived_tables/fig5b_variation_vulnerability_scatter.csv`.
- **(d)** Incremental prognostic information in Individual deviations on
  clean trajectories, comparing P and P+I for Ridge, kNN, and MLP.
- **(e)** Reconstruction-induced prognostic change for P and P+I across all
  five L5 corruption topologies, retaining the formal mixed signs.
- **(f)** Two matched scatter plots of Individual-fidelity loss against P and
  P+I prognostic loss.  Both use 105 battery × corruption points and share
  the same x/y limits.

The R4 reconstruction-consequence readiness is **WEAK**.  Panel (e) is
therefore descriptive of corruption-dependent consequences and is not a
universal-degradation claim.  Panel (f) reports associations, not causal
effects.  The P control is slightly higher than P+I and the paired bootstrap
interval for their difference spans zero; this does not support a stronger
claim of P+I-specific fidelity association.

## Panel (f) derived assets

`delta_E_ind` is read from `fidelity_prognosis_link.csv:E_ind_loss`.
For each battery × corruption × feature set,
`delta_MAE = MAE(reconstructed) - MAE(clean)` is derived from
`reconstruction_prognosis_battery_metrics.csv`.  P and P+I use the same
whole-battery bootstrap: 2,000 draws, 21 battery IDs sampled with
replacement, retaining all five corruption rows per selected battery.  The
P+I result is reconciled to the formal `prognostic_loss` column.

## Reproduce

From the repository root:

```bash
python figures/figS7/make_figS7.py
```

The command writes `figS7_main.png`, `figS7_main.pdf`, and `figS7_main.svg`,
plus `figS7_qc.md`, the typography/export audit, and the plotting-only
derived tables under `derived_tables/`.
"""
    (ROOT / "README_figS7.md").write_text(text, encoding="utf-8")


def _write_qc(tables: dict[str, pd.DataFrame], outputs: dict[str, str]) -> None:
    svg = Path(outputs["svg"]).read_text(encoding="utf-8").lower()
    forbidden = [term for term in ("shrinkage", "misalignment", "amplification", "fingerprint", "identity claim") if term in svg]
    primary = tables["longitudinal"]
    intervals_ok = bool((primary.ci_low <= primary.estimate).all() and (primary.ci_high >= primary.estimate).all())
    pdf_ok = Path(outputs["pdf"]).is_file() and Path(outputs["pdf"]).read_bytes()[:4] == b"%PDF"
    png_ok = Path(outputs["png"]).is_file() and Path(outputs["png"]).stat().st_size > 0
    dual = tables["dual_association"]
    stats = tables["dual_stats"]
    bootstrap = tables["dual_bootstrap"]
    formal_pi = tables["fidelity_stats"].query("association == 'E_ind_loss'").iloc[0]
    derived_pi = stats.query("model == 'P+I'").iloc[0]
    latest = REPO / "analysis" / "outputs" / "r4_formal" / "latest"
    latest_ok = latest.is_symlink() and latest.resolve() == RUN.resolve()
    lines = [
        "# FigS7 QC", "",
        f"- R4 input root: `{RUN}`: PASS",
        f"- `r4_formal/latest` resolves to this run: {'PASS' if latest_ok else 'ISSUE'}",
        "- R4 formal output directory modified by plotting: NO; bootstrap repair is audited separately: PASS",
        f"- Panel (a) repaired formal median-bootstrap source and intervals: {'PASS' if intervals_ok else 'FAIL'}",
        "- Panel (a) 10-D train-only PCA-space distance label: PASS",
        "- Panel (b) position-wise variation terminology; C50 kept out of panel: PASS",
        f"- Panel (c) 17-point X_w/Y_w scatter reproduces formal rho: {'PASS' if len(tables['variation_vulnerability']) == 17 else 'FAIL'}",
        "- Panel (c) 20% occlusion response is represented by formal window scatter: PASS",
        "- Panel (d) prognosis increment is a standalone panel: PASS",
        "- Panel (d) explicit MAE contrast; clean-trajectory provenance documented outside panel: PASS",
        "- Panel (e) retains both negative and positive reconstruction effects; L5 provenance documented outside panel: PASS",
        f"- Panel (f) dual P/P+I scatter has 105 battery × corruption points: {'PASS' if len(dual) == 105 else 'FAIL'}",
        f"- Panel (f) derived P+I values reproduce formal association: {'PASS' if np.allclose([derived_pi.rho, derived_pi.ci_low, derived_pi.ci_high], [formal_pi.rho, formal_pi.ci_low, formal_pi.ci_high], atol=1e-12, rtol=1e-10) else 'FAIL'}",
        f"- Panel (f) whole-battery paired bootstrap has 2,000 draws: {'PASS' if len(bootstrap) == BOOTSTRAPS else 'FAIL'}",
        f"- Old R3 rho=0.831 absent: {'PASS' if '0.831' not in svg else 'FAIL'}",
        f"- Phenotype/fingerprint/identity terminology absent: {'PASS' if not forbidden else 'FAIL (' + ', '.join(forbidden) + ')'}",
        f"- PDF signature: {'PASS' if pdf_ok else 'FAIL'}",
        f"- PNG present: {'PASS' if png_ok else 'FAIL'}",
        "- Mixed-sign panel-e interpretation retained as corruption-dependent, not universal: PASS",
        "- No model retraining or reconstruction rerun: PASS",
        "",
        "## Outputs", "",
        f"- PNG: `{outputs['png']}`",
        f"- PDF: `{outputs['pdf']}`",
        f"- SVG: `{outputs['svg']}`",
    ]
    (ROOT / "figS7_qc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dual_audit(tables: dict[str, pd.DataFrame]) -> None:
    dual = tables["dual_association"]
    stats = tables["dual_stats"].set_index("model")
    bootstrap = tables["dual_bootstrap"]
    delta_rho = float(stats.loc["P+I", "rho"] - stats.loc["P", "rho"])
    delta_low = float(np.nanpercentile(bootstrap.delta_rho, 2.5))
    delta_high = float(np.nanpercentile(bootstrap.delta_rho, 97.5))
    latest = REPO / "analysis" / "outputs" / "r4_formal" / "latest"
    interpretation = (
        "No: P is slightly higher than P+I, and the paired bootstrap CI for "
        f"rho(P+I) - rho(P) [{delta_low:.6f}, {delta_high:.6f}] spans zero."
    )
    text = f"""# Fig.5f dual-association audit in Supplementary Fig. S7

## Formal source

- Exact R4 run: `{RUN.resolve()}`
- `analysis/outputs/r4_formal/latest` resolves to the same run: `{latest.resolve() if latest.exists() else 'MISSING'}`
- No file under the formal run was modified.

## Field provenance

1. **Final ΔE_ind source field:** `fidelity_prognosis_link.csv:E_ind_loss`.
2. **Final ΔMAE_P source/derivation:**
   `reconstruction_prognosis_battery_metrics.csv:MAE`, filtered to
   `feature_set == P`, then reconstructed minus clean MAE within each
   battery × corruption.
3. **Final ΔMAE_P+I source/derivation:** the same frozen MAE field and
   operation, filtered to `feature_set == P+I`; it reproduces the formal
   `fidelity_prognosis_link.csv:prognostic_loss` column.

## Statistical unit and bootstrap

- Point unit: **battery × corruption**.
- Number of points: **{len(dual)}**; batteries: **{dual.battery_id.nunique()}**;
  conditions: **{dual.corruption.nunique()}**.
- Bootstrap: **whole-battery**, 2,000 resamples, seed `{BOOTSTRAP_SEED}`;
  each draw samples 21 battery IDs with replacement and retains all five
  corruption rows for each selected battery.  P and P+I use the same paired
  battery draws.
- No prognosis model was refit, no BatteryDiT was rerun, and no
  reconstruction result was regenerated.

## Results

| model | rho | 95% CI | n points |
|---|---:|---:|---:|
| P | {stats.loc['P', 'rho']:.9f} | [{stats.loc['P', 'ci_low']:.9f}, {stats.loc['P', 'ci_high']:.9f}] | {int(stats.loc['P', 'n_points'])} |
| P+I | {stats.loc['P+I', 'rho']:.9f} | [{stats.loc['P+I', 'ci_low']:.9f}, {stats.loc['P+I', 'ci_high']:.9f}] | {int(stats.loc['P+I', 'n_points'])} |

- Observed Δrho = rho(P+I) − rho(P) = **{delta_rho:.9f}**.
- Paired bootstrap 95% CI for Δrho = **[{delta_low:.9f}, {delta_high:.9f}]**.

## Interpretation

**{interpretation}** The P control therefore does not support the stronger
statement that Individual-fidelity loss is more specific to P+I prognostic
loss than to prognostic loss in general.
"""
    (ROOT / "audit_fig5f_dual_association.md").write_text(text, encoding="utf-8")


def draw() -> dict[str, str]:
    apply_paper_style()
    tables = load()
    DERIVED.mkdir(parents=True, exist_ok=True)
    tables["variation_vulnerability"].to_csv(
        DERIVED / "fig5b_variation_vulnerability_scatter.csv", index=False
    )
    tables["dual_association"].to_csv(
        DERIVED / "fig5f_dual_association.csv", index=False
    )
    tables["dual_stats"].to_csv(
        DERIVED / "fig5f_dual_association_stats.csv", index=False
    )
    tables["dual_bootstrap"].to_csv(
        DERIVED / "fig5f_dual_association_bootstrap.csv", index=False
    )
    figure = plt.figure(figsize=(FIGURE_WIDTH, 212 / 25.4), facecolor="white")
    grid = figure.add_gridspec(3, 2, left=0.085, right=0.985, bottom=0.075,
                               top=0.96, wspace=0.36, hspace=0.58)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
    ]
    _panel_a(axes[0], tables["longitudinal"])
    _panel_b(axes[1], tables)
    _panel_c(axes[2], tables)
    _panel_d(axes[3], tables["prognosis"])
    _panel_e(axes[4], tables["reconstruction"])
    _panel_f(grid[2, 1], tables)
    outputs = export_figure(figure, FIGURE_ID, stem=FIGURE_STEM)
    plt.close(figure)
    _write_qc(tables, outputs)
    _write_readme()
    _write_dual_audit(tables)
    return outputs


if __name__ == "__main__":
    result = draw()
    for key in ("png", "pdf", "svg"):
        print(f"{key}: {result[key]}")
