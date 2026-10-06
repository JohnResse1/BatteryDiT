"""Regression contract for the formal R4 longitudinal bootstrap estimator."""
from __future__ import annotations

import numpy as np

from analysis.workflows.r4_formal_canonical import _longitudinal_bootstrap_statistic


def test_longitudinal_median_estimator_is_not_mean() -> None:
    battery_level_values = np.asarray([1.0, 2.0, 3.0, 100.0])

    observed = _longitudinal_bootstrap_statistic(battery_level_values, "median")

    assert observed == 2.5
    assert observed != float(np.mean(battery_level_values))
