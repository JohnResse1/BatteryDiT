"""Regression test for HUST charge-stage selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_preprocessing.sources.datasets.hust import _charge_curve


def main() -> None:
    frame = pd.DataFrame(
        {
            "Status": [
                "Constant current charge",
                "Constant current-constant voltage charge",
                "Constant current discharge_0",
                "Constant current discharge_1",
            ],
            "Current (mA)": [2000.0, 500.0, -1000.0, -2000.0],
            "Voltage (V)": [3.0, 3.6, 3.4, 2.0],
            "Time (s)": [0.0, 1.0, 2.0, 3.0],
        }
    )
    current, voltage, time = _charge_curve(frame)
    np.testing.assert_array_equal(current, [2.0, 0.5])
    np.testing.assert_array_equal(voltage, [3.0, 3.6])
    np.testing.assert_array_equal(time, [0.0, 1.0])
    print("HUST charge-stage selection regression passed")


if __name__ == "__main__":
    main()
