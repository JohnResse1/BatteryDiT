"""Materialize canonical plotting assets for the current Fig. 4.

This helper performs only plotting-table export and a forward evaluation of
the already-certified MIT SOH-conditioned Ridge coefficients for the visual
population-curve family.  It does not fit a model, run inference, or compute
any formal statistic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
DATA_DIR = FIGURE_DIR / "data"

WORKFLOW_SOURCE = REPO / "analysis/outputs/population_individual_workflow_example.npz"
CONDITION_SOURCE = REPO / (
    "paper_results/table_s6_population_individual/"
    "population_individual_condition_summary.csv"
)
SOH_LINKAGE_SOURCE = REPO / "analysis/outputs/population_individual_soh_linkage.csv"
SIGNATURE_LINKAGE_SOURCE = REPO / "analysis/outputs/population_individual_signature_linkage.csv"
MIT_RAW_SOURCE = REPO / "paper_results/fig5/raw/cycle_level_decomposition_mit.csv"
MIT_PROVENANCE_ROOT = REPO / "analysis/outputs/canonical/population_individual_provenance"
MIT_TRAIN_CLEAN_ROOT = REPO / "source_code/data/mit/train/clean"

# The projection is a visualization-only display asset.  It is deliberately
# derived from clean training trajectories and the already-frozen Ridge
# parameters; it is not used by any metric or formal analysis.
MANIFOLD_DISPLAY_POINTS_PER_BATTERY = 160

DATASET_ORDER = ["MIT", "HUST", "MICH", "Stanford", "TJU-NCA", "TJU-NCM"]
CONDITION_ORDER = ["downsample", "random_missing", "segment_missing", "destructive", "hybrid"]
CONDITION_LABELS = {
    "downsample": "Downsampling",
    "random_missing": "Random",
    "segment_missing": "Segment",
    "destructive": "Destructive",
    "hybrid": "Hybrid",
}


def _spline_design(values: np.ndarray, knots: np.ndarray, degree: int,
                   lower: float, upper: float) -> np.ndarray:
    """Evaluate the saved sklearn-compatible B-spline basis with NumPy.

    The formal Ridge coefficients and the full knot vector are already saved
    in the certified provenance asset.  Re-evaluating that fixed basis here
    avoids fitting or selecting anything in the figure layer and keeps the
    plotting-asset exporter free of a scikit-learn runtime dependency.
    ``include_bias=False`` is implemented by dropping the final basis column,
    matching the frozen `SplineTransformer` configuration.
    """
    x = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), lower, upper)
    t = np.asarray(knots, dtype=np.float64).reshape(-1)
    if len(t) <= degree + 1:
        raise ValueError("saved spline knot vector is too short")
    n0 = len(t) - 1
    basis = np.zeros((len(x), n0), dtype=np.float64)
    for index in range(n0):
        basis[:, index] = ((x >= t[index]) & (x < t[index + 1])).astype(float)
    basis[x == t[-1], -1] = 1.0
    for order in range(1, degree + 1):
        reduced = np.zeros((len(x), n0 - order), dtype=np.float64)
        for index in range(n0 - order):
            left_denominator = t[index + order] - t[index]
            right_denominator = t[index + order + 1] - t[index + 1]
            if left_denominator:
                reduced[:, index] += (
                    (x - t[index]) / left_denominator * basis[:, index]
                )
            if right_denominator:
                reduced[:, index] += (
                    (t[index + order + 1] - x) / right_denominator
                    * basis[:, index + 1]
                )
        basis = reduced
    return basis[:, :-1]


def _latest_mit_parameters() -> Path:
    candidates = []
    for path in MIT_PROVENANCE_ROOT.glob("*/provenance/population_manifold_parameters.npz"):
        config = path.parent / "analysis_config.json"
        if not config.is_file() or path.parent.name != "provenance":
            continue
        payload = json.loads(config.read_text(encoding="utf-8"))
        if payload.get("dataset") == "mit" and "debug_smoke" not in path.parts:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No formal MIT population-manifold parameters were found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _check_inputs() -> Path:
    required = [
        WORKFLOW_SOURCE,
        CONDITION_SOURCE,
        SOH_LINKAGE_SOURCE,
        SIGNATURE_LINKAGE_SOURCE,
        MIT_RAW_SOURCE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing formal Fig.4 inputs:\n" + "\n".join(missing))
    return _latest_mit_parameters()


def _write_workflow_assets(parameters_path: Path) -> dict[str, object]:
    with np.load(WORKFLOW_SOURCE, allow_pickle=False) as archive:
        clean = np.asarray(archive["clean"], dtype=np.float64)
        reference = np.asarray(archive["reference"], dtype=np.float64)
        population = np.asarray(archive["clean_population"], dtype=np.float64)
        individual = np.asarray(archive["clean_individual"], dtype=np.float64)
        key = str(archive["key"])
        true_soh = float(archive["true_soh"])
        h_rec = float(archive["h_rec"])

    degradation = clean - reference
    if not np.allclose(degradation, population + individual, atol=2e-6, rtol=2e-6):
        raise RuntimeError("The certified workflow example does not satisfy D=M+R")

    battery_id, cycle_name = key.rsplit("/", 1)
    cycle_id = int(cycle_name.removesuffix(".npz"))
    progress = np.linspace(0.0, 1.0, len(clean))
    workflow = pd.DataFrame({
        "progress": progress,
        "V_clean": clean,
        "V_ref": reference,
        "D_clean": degradation,
        "M_clean": population,
        "R_clean": individual,
        "battery_id": battery_id,
        "cycle_id": cycle_id,
        "soh": true_soh,
        "h_rec": h_rec,
    })
    workflow.to_csv(DATA_DIR / "fig4a_workflow_curves.csv", index=False)

    with np.load(parameters_path, allow_pickle=True) as archive:
        n_knots = int(archive["spline_n_knots"])
        degree = int(archive["spline_degree"])
        knots = np.asarray(archive["spline_knots"], dtype=np.float64).reshape(-1)
        train_min = float(archive["train_soh_min"])
        train_max = float(archive["train_soh_max"])
        coefficients = np.asarray(archive["ridge_coefficients"], dtype=np.float64)
        intercept = np.asarray(archive["ridge_intercept"], dtype=np.float64)

    # Recreate only the saved spline basis.  The Ridge coefficients and knots
    # are loaded from certified provenance; no fitting or selection happens.
    if len(knots) != n_knots + 2 * degree + 0:
        raise RuntimeError(
            f"saved spline knot vector has length {len(knots)}; expected "
            f"{n_knots + 2 * degree}"
        )
    family_soh = list(np.linspace(train_min, train_max, 5))
    if not any(np.isclose(true_soh, value, atol=1e-7) for value in family_soh):
        family_soh.append(true_soh)
    family_soh = sorted(family_soh)
    design = _spline_design(family_soh, knots, degree, train_min, train_max)
    if design.shape[1] != coefficients.shape[1]:
        raise RuntimeError(
            f"saved spline basis has {design.shape[1]} columns but Ridge "
            f"coefficients have {coefficients.shape[1]}"
        )
    family_values = design @ coefficients.T + intercept[None, :]
    current_index = int(np.argmin(np.abs(np.asarray(family_soh) - true_soh)))
    if not np.allclose(family_values[current_index], population, atol=2e-5, rtol=2e-5):
        raise RuntimeError("Saved MIT Ridge coefficients do not reproduce M_clean")

    family_rows = []
    for index, (soh, values) in enumerate(zip(family_soh, family_values)):
        role = "current_sample" if np.isclose(soh, true_soh, atol=1e-7) else f"SOH_grid_{index + 1}"
        for x, value in zip(progress, values):
            family_rows.append({
                "progress": float(x),
                "soh": float(soh),
                "M_population": float(value),
                "curve_role": role,
            })
    pd.DataFrame(family_rows).to_csv(DATA_DIR / "fig4a_population_family.csv", index=False)
    return {
        "source": str(WORKFLOW_SOURCE.relative_to(REPO)),
        "parameters": str(parameters_path.relative_to(REPO)),
        "battery_id": battery_id,
        "cycle_id": cycle_id,
        "soh": true_soh,
        "h_rec": h_rec,
        "ridge_n_knots": n_knots,
        "ridge_degree": degree,
        "ridge_train_soh_min": train_min,
        "ridge_train_soh_max": train_max,
    }


def _saved_population_predictor(parameters_path: Path):
    """Return a predictor backed only by the saved canonical Ridge state."""
    with np.load(parameters_path, allow_pickle=True) as archive:
        degree = int(archive["spline_degree"])
        knots = np.asarray(archive["spline_knots"], dtype=np.float64).reshape(-1)
        train_min = float(archive["train_soh_min"])
        train_max = float(archive["train_soh_max"])
        coefficients = np.asarray(archive["ridge_coefficients"], dtype=np.float64)
        intercept = np.asarray(archive["ridge_intercept"], dtype=np.float64)

    def predict(values: np.ndarray) -> np.ndarray:
        design = _spline_design(values, knots, degree, train_min, train_max)
        if design.shape[1] != coefficients.shape[1]:
            raise RuntimeError(
                f"saved spline basis has {design.shape[1]} columns but Ridge "
                f"coefficients have {coefficients.shape[1]}"
            )
        return design @ coefficients.T + intercept[None, :]

    return predict, train_min, train_max


def _write_manifold_projection_asset(parameters_path: Path) -> dict[str, object]:
    """Export the old Fig. 5-style real manifold projection for Fig. 4a.

    The display basis is isolated from the test example: its first direction
    follows the endpoints of the frozen population trajectory family and its
    second direction is the leading clean-training residual direction after
    orthogonalisation.  The projection is only for the panel-a visual cue;
    formal metrics remain defined in the original 1,000-point trajectory
    space.
    """
    if not MIT_TRAIN_CLEAN_ROOT.is_dir():
        raise FileNotFoundError(MIT_TRAIN_CLEAN_ROOT)

    # Use the certified loader only to read the existing clean training files
    # and apply the already-frozen early-reference subtraction.  No manifold
    # fit, model selection, inference, or statistic is run here.
    import sys

    source_code = REPO / "source_code"
    if str(source_code) not in sys.path:
        sys.path.insert(0, str(source_code))
    from exp import population_individual as certified  # noqa: PLC0415

    train = certified.load_clean("mit", "train")
    train_degradation, _, _ = certified.residualize(train, require_five=True)
    train_degradation = np.asarray(train_degradation, dtype=np.float64).reshape(len(train_degradation), -1)
    train_soh = np.asarray(train["soh"], dtype=np.float64)
    train_cells = np.asarray(train["cells"]).astype(str)
    train_cycles = np.asarray(train["cycles"], dtype=int)

    predict, train_min, train_max = _saved_population_predictor(parameters_path)
    manifold_soh = np.linspace(train_min, train_max, 500, dtype=np.float64)
    manifold = predict(manifold_soh)
    train_population = predict(train_soh)

    stage_axis = manifold[-1] - manifold[0]
    stage_norm = float(np.linalg.norm(stage_axis))
    if not np.isfinite(stage_norm) or stage_norm <= 1e-12:
        raise RuntimeError("cannot define population-aligned display axis")
    stage_axis = stage_axis / stage_norm

    train_residual = train_degradation - train_population
    residual_parallel = train_residual @ stage_axis
    residual_orthogonal = train_residual - residual_parallel[:, None] * stage_axis[None, :]
    centered_orthogonal = residual_orthogonal - residual_orthogonal.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered_orthogonal, full_matrices=False)
    individual_axis = np.asarray(right_vectors[0], dtype=np.float64)
    individual_axis -= float(np.dot(individual_axis, stage_axis)) * stage_axis
    individual_norm = float(np.linalg.norm(individual_axis))
    if not np.isfinite(individual_norm) or individual_norm <= 1e-12:
        raise RuntimeError("cannot define orthogonal individual display axis")
    individual_axis /= individual_norm
    score = centered_orthogonal @ individual_axis
    if float(np.percentile(score, 5) + np.percentile(score, 95)) < 0.0:
        individual_axis = -individual_axis

    display_center = manifold.mean(axis=0)

    def project(values: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        centered = values - display_center[None, :]
        return np.column_stack((centered @ stage_axis, centered @ individual_axis))

    projected_training = project(train_degradation)
    projected_manifold = project(manifold)

    with np.load(WORKFLOW_SOURCE, allow_pickle=False) as archive:
        workflow_key = str(archive["key"])
        current_degradation = np.asarray(archive["clean"], dtype=np.float64) - np.asarray(archive["reference"], dtype=np.float64)
        current_population = np.asarray(archive["clean_population"], dtype=np.float64)
        current_soh = float(archive["true_soh"])
    current_battery, current_cycle_name = workflow_key.rsplit("/", 1)
    current_cycle = int(current_cycle_name.removesuffix(".npz"))
    current_points = project(np.stack((current_population, current_degradation)))

    # Keep the cloud legible and the vector export compact while preserving a
    # deterministic life-course sample from every clean training battery.
    selected_indices: list[int] = []
    for battery in sorted(set(train_cells)):
        indices = np.flatnonzero(train_cells == battery)
        n_selected = min(MANIFOLD_DISPLAY_POINTS_PER_BATTERY, len(indices))
        positions = np.linspace(0, len(indices) - 1, n_selected, dtype=int)
        selected_indices.extend(indices[np.unique(positions)].tolist())
    selected_indices = sorted(set(selected_indices))

    rows: list[dict[str, object]] = []
    for index in selected_indices:
        rows.append({
            "display_role": "training_cloud",
            "point_index": int(index),
            "projection_x": float(projected_training[index, 0]),
            "projection_y": float(projected_training[index, 1]),
            "soh": float(train_soh[index]),
            "battery_id": str(train_cells[index]),
            "cycle_id": int(train_cycles[index]),
        })
    for index, (soh, point) in enumerate(zip(manifold_soh, projected_manifold)):
        rows.append({
            "display_role": "population_manifold",
            "point_index": int(index),
            "projection_x": float(point[0]),
            "projection_y": float(point[1]),
            "soh": float(soh),
            "battery_id": "",
            "cycle_id": -1,
        })
    for role, point in zip(("current_population", "current_sample"), current_points):
        rows.append({
            "display_role": role,
            "point_index": -1,
            "projection_x": float(point[0]),
            "projection_y": float(point[1]),
            "soh": current_soh,
            "battery_id": current_battery,
            "cycle_id": current_cycle,
        })

    pd.DataFrame(rows).to_csv(DATA_DIR / "fig4a_manifold_projection.csv", index=False)
    return {
        "source_training_data": str(MIT_TRAIN_CLEAN_ROOT.relative_to(REPO)),
        "source_parameters": str(parameters_path.relative_to(REPO)),
        "training_rows": int(len(train_degradation)),
        "display_training_rows": int(len(selected_indices)),
        "manifold_rows": int(len(manifold_soh)),
        "display_points_per_battery": MANIFOLD_DISPLAY_POINTS_PER_BATTERY,
        "projection_definition": "stage axis = frozen population-manifold endpoint direction; individual axis = leading clean-training residual direction orthogonalized to stage axis",
        "projection_center": "mean of the 500-point frozen population manifold grid",
        "metrics_evaluated_in": "original 1,000-point trajectory space",
        "example_battery_id": current_battery,
        "example_cycle_id": current_cycle,
        "example_soh": current_soh,
    }


def _write_recoverability_assets() -> dict[str, object]:
    table = pd.read_csv(CONDITION_SOURCE)
    expected = len(CONDITION_ORDER) * 5
    table = table[table.dataset.eq("MIT") & table.condition.isin(CONDITION_ORDER)].copy()
    if len(table) != expected:
        raise RuntimeError(f"Expected {expected} formal MIT condition rows, found {len(table)}")
    if table.duplicated(["dataset", "condition", "severity"]).any():
        raise RuntimeError("Formal recoverability table contains duplicate cells")
    table["corruption_type"] = table["condition"]
    table["corruption_label"] = table["condition"].map(CONDITION_LABELS)
    table["severity_level"] = table["severity"]
    table["E_pop_median"] = table["median_E_pop"]
    table["E_ind_median"] = table["median_E_ind"]
    table["C_ind_median"] = table["median_C_ind"]
    table["Q_median"] = table["median_Q"]
    table = table.sort_values(
        ["dataset", "condition", "severity_index"],
        key=lambda values: values.map({name: index for index, name in enumerate(DATASET_ORDER)})
        if values.name == "dataset" else values,
        kind="stable",
    )
    columns = [
        "dataset", "condition", "corruption_type", "corruption_label", "severity",
        "severity_level", "severity_index", "n_batteries", "n_cycles",
        "E_pop_median", "E_ind_median", "C_ind_median", "Q_median",
        "median_signature_error", "median_soh_error", "median_waveform_rmse",
    ]
    table[columns].to_csv(DATA_DIR / "fig4b_recoverability.csv", index=False)
    return {
        "source": str(CONDITION_SOURCE.relative_to(REPO)),
        "datasets": ["MIT"],
        "conditions": [CONDITION_LABELS[name] for name in CONDITION_ORDER],
        "rows": int(len(table)),
        "aggregation": "formal battery-balanced median",
    }


def _write_scatter_assets() -> dict[str, object]:
    raw = pd.read_csv(MIT_RAW_SOURCE)
    required = {
        "cell", "condition", "severity", "severity_index", "cycle",
        "E_ind", "soh_error", "C_ind", "signature_error",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Formal MIT scatter source is missing columns: {sorted(missing)}")
    grouped = raw.groupby(
        ["cell", "condition", "severity", "severity_index"], as_index=False,
    ).agg(
        E_ind=("E_ind", "median"),
        SOH_error=("soh_error", "median"),
        C_ind=("C_ind", "median"),
        signature_error=("signature_error", "median"),
        sample_count=("cycle", "size"),
    )
    if len(grouped) != 21 * len(CONDITION_ORDER) * 5:
        raise RuntimeError(f"Expected 525 formal MIT scatter units, found {len(grouped)}")
    grouped["condition_order"] = grouped["condition"].map(
        {name: index for index, name in enumerate(CONDITION_ORDER)}
    )
    grouped = grouped.sort_values(
        ["cell", "condition_order", "severity_index"], kind="stable"
    ).rename(columns={
        "cell": "battery_id",
        "condition": "corruption_type",
        "severity": "severity_level",
    }).drop(columns="condition_order")
    scatter_columns = [
        "battery_id", "corruption_type", "severity_level", "E_ind",
        "SOH_error", "C_ind", "signature_error", "sample_count",
    ]
    grouped[scatter_columns].to_csv(DATA_DIR / "fig4c1_eind_vs_soh.csv", index=False)
    grouped[scatter_columns].to_csv(DATA_DIR / "fig4c2_cind_vs_signature.csv", index=False)

    soh = pd.read_csv(SOH_LINKAGE_SOURCE)
    signature = pd.read_csv(SIGNATURE_LINKAGE_SOURCE)
    soh = soh[soh.dataset.eq("MIT")].copy()
    signature = signature[signature.dataset.eq("MIT")].copy()
    if len(soh) != 1 or len(signature) != 1:
        raise RuntimeError("Formal MIT relevance linkages must contain one row each")
    soh_row = soh.iloc[0]
    signature_row = signature.iloc[0]
    stats = {
        "aggregation_unit": "one battery × one corruption-severity condition",
        "bootstrap_unit": "whole_battery",
        "bootstrap_resamples": 2000,
        "n_points": int(len(grouped)),
        "n_batteries": int(grouped.battery_id.nunique()),
        "E_ind_vs_SOH_error": {
            "rho": float(soh_row.spearman_rho),
            "ci_low": float(soh_row.bootstrap_low),
            "ci_high": float(soh_row.bootstrap_high),
            "n_batteries": int(soh_row.n_batteries),
        },
        "C_ind_vs_signature_error": {
            "rho": float(signature_row.spearman_rho),
            "ci_low": float(signature_row.bootstrap_low),
            "ci_high": float(signature_row.bootstrap_high),
            "n_batteries": int(signature_row.n_batteries),
        },
    }
    (DATA_DIR / "fig4c_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "raw_source": str(MIT_RAW_SOURCE.relative_to(REPO)),
        "soh_stats_source": str(SOH_LINKAGE_SOURCE.relative_to(REPO)),
        "signature_stats_source": str(SIGNATURE_LINKAGE_SOURCE.relative_to(REPO)),
        "rows": int(len(grouped)),
        "aggregation": "median within each formal battery × corruption-severity condition",
        "bootstrap_unit": "whole_battery",
        "bootstrap_resamples": 2000,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parameters = _check_inputs()
    provenance = {
        "figure": "Fig.4 V1",
        "workflow": _write_workflow_assets(parameters),
        "manifold_projection": _write_manifold_projection_asset(parameters),
        "recoverability": _write_recoverability_assets(),
        "relevance_scatter": _write_scatter_assets(),
        "old_repo_runtime_dependency": False,
        "statistics_recomputed": False,
    }
    (DATA_DIR / "fig4_asset_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Fig.4 plotting assets written -> {DATA_DIR}")


if __name__ == "__main__":
    main()
