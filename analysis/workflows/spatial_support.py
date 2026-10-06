"""Formal spatial-support helpers shared by the R4 workflows.

The functions in this module are small, deterministic transformations used by
the canonical spatial-support and sliding-occlusion workflows.  They do not
run an experiment or write output files; workflow-specific execution remains
in the canonical modules under :mod:`analysis.workflows`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


WINDOW_WIDTHS = (0.10, 0.20, 0.30)
WINDOW_STRIDE = 0.05
MASK_VALUE = 0.0


def clean_spatial_support(sequences, grid_points: int = 500) -> pd.DataFrame:
    """Return population and individual support on a common trajectory grid.

    Population support is the pointwise variance of the frozen nominal-SOH
    manifold.  Individual support is the median, across test batteries, of
    each battery's cycle-wise median squared residual.  No test-SOH sampling
    distribution enters either quantity.
    """
    state = sequences["state"]
    certified = state["certified"]
    train = certified.load_clean("mit", "train")
    soh_grid = np.linspace(train["soh"].min(), train["soh"].max(), grid_points)
    population = np.asarray(
        certified.predict(state["model"], state["spline"], soh_grid),
        dtype=np.float64,
    ).reshape(grid_points, -1)
    population_support = population.var(axis=0)

    per_battery = []
    for sequence in sequences["test"]:
        per_battery.append(np.median(sequence["residual"] ** 2, axis=0))
    individual_support = np.median(np.stack(per_battery), axis=0)

    return pd.DataFrame(
        {
            "position": np.arange(population_support.size),
            "normalized_progress": np.linspace(0, 1, population_support.size),
            "P_pop": population_support,
            "P_ind": individual_support,
            "n_test_batteries": len(per_battery),
        }
    )


def concentration(support: np.ndarray, fractions=(0.5, 0.8, 0.9)) -> dict:
    """Summarize how concentrated a support profile is."""
    support = np.asarray(support, float)
    order = np.argsort(support)[::-1]
    cumulative = np.cumsum(support[order]) / support.sum()
    out = {}
    for fraction in fractions:
        out[f"support_fraction_{int(fraction * 100)}"] = float(
            (np.searchsorted(cumulative, fraction) + 1) / support.size
        )
    probability = support / support.sum()
    positive = probability[probability > 0]
    out["entropy_effective_support"] = float(
        np.exp(-(positive * np.log(positive)).sum()) / support.size
    )
    out["participation_ratio_fraction"] = float(
        (support.sum() ** 2 / (support**2).sum()) / support.size
    )
    return out


def window_definitions(length: int, widths=WINDOW_WIDTHS) -> list[dict]:
    """Return contiguous zero-occlusion windows for each requested width."""
    definitions = []
    for width in widths:
        span = int(round(width * length))
        start = 0.0
        while start + width <= 1.0 + 1e-9:
            start_index = int(round(start * length))
            end = min(length, start_index + span)
            definitions.append(
                {
                    "window_id": f"w{len(definitions):02d}_{start:.2f}",
                    "width_fraction": float(width),
                    "window_start": float(start),
                    "window_end": float(start + width),
                    "start_index": start_index,
                    "end_index_exclusive": end,
                    "width_points": int(end - start_index),
                }
            )
            start = round(start + WINDOW_STRIDE, 6)
    return definitions


def occlusion_mask(length: int, start: int, end: int) -> np.ndarray:
    """Return the shared zero mask used for voltage and current."""
    mask = np.ones(length, dtype=np.float32)
    mask[start:end] = MASK_VALUE
    return mask
