"""Regression contract for the formal Part 4 manifold gate and bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from exp.population_individual import bootstrap_battery_median, select_manifold


def _split(prefix: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cells = np.repeat(np.asarray([f"{prefix}{index}" for index in range(4)]), 8)
    cycles = np.tile(np.arange(1, 9), 4)
    soh = 1.0 - 0.01 * cycles + rng.normal(0.0, 0.001, len(cycles))
    coordinate = np.linspace(0.0, 1.0, 1000, dtype=np.float32)[None, :, None]
    voltage = 3.2 - 0.4 * (1.0 - soh)[:, None, None] * coordinate
    voltage += rng.normal(0.0, 1e-4, voltage.shape)
    return {
        "voltage": voltage.astype(np.float32),
        "soh": soh.astype(np.float32),
        "cells": cells,
        "cycles": cycles,
    }


def main() -> None:
    _, _, scale, table, summary = select_manifold(_split("train", 7), _split("vali", 11))
    assert np.isfinite(scale) and scale > 0.0
    assert len(table[table.row_type == "candidate"]) == 18
    assert len(table[table.row_type == "selected_battery"]) == 4
    assert {"manifold_gate_passed", "bootstrap_selected_minus_constant_low"} <= set(summary)

    frame = pd.DataFrame({"cell": ["a", "a", "b", "b"], "metric": [1.0, 2.0, 3.0, 4.0]})
    estimate, low, high = bootstrap_battery_median(frame, "metric", seed_offset=1)
    assert np.isfinite([estimate, low, high]).all()
    assert low <= estimate <= high
    print("population-individual protocol regression passed")


if __name__ == "__main__":
    main()
