"""Regression test for train-only OCV prior extraction."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from models.ocv.extraction import (
    OCVPriorPolicy,
    save_training_ocv,
    select_ocv_source,
    training_cells_from_directory,
)
from models.BatteryDiT import Model as BatteryDiT


def write_curve(path: Path, current: float, voltage_start: float, voltage_span: float = 0.8) -> None:
    points = 30
    voltage = np.linspace(voltage_start, voltage_start + voltage_span, points)
    voltage[12] -= 0.1  # The artifact must become monotone after construction.
    np.savez(path, I=np.full((points, 1), current), V=voltage[:, None])


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        train = root / "train"
        test = root / "test"
        for cell, current, start in [("slow_train", 1.0, 3.0), ("fast_train", 3.0, 3.2)]:
            directory = train / cell
            directory.mkdir(parents=True)
            for cycle in range(8, 13):
                write_curve(directory / f"{cycle}.npz", current, start)
        # This cell has the lowest current but lacks most of the population's
        # voltage span, so it must not silently define the low-SOC boundary.
        truncated = train / "truncated_slow_train"
        truncated.mkdir()
        for cycle in range(8, 13):
            write_curve(truncated / f"{cycle}.npz", 0.5, 3.65, voltage_span=0.25)
        test.mkdir()
        write_curve(test / "leaked_test.npz", 0.01, 4.0)
        cells = training_cells_from_directory(train)
        selected = select_ocv_source(cells, OCVPriorPolicy(soc_points=32))
        assert selected.cycle == 10
        assert np.allclose(selected.current, 1.0)
        artifact_path = save_training_ocv(train, root / "artifacts" / "toy_ocv.npz", OCVPriorPolicy(soc_points=32))
        with np.load(artifact_path) as artifact:
            assert artifact["soc"].shape == artifact["voltage"].shape == (32,)
            assert np.all(np.diff(artifact["voltage"]) >= 0)
            expected_soc = artifact["soc"].copy()
            expected_voltage = artifact["voltage"].copy()
        metadata = __import__("json").loads(artifact_path.with_suffix(".json").read_text())
        assert metadata["selected_cell"] == "slow_train"
        assert metadata["selected_cycle"] == 10
        model = BatteryDiT(
            SimpleNamespace(
                v_min=2.0,
                v_max=4.0,
                po_type="MIT",
                ocv_table_path=str(artifact_path),
            )
        )
        np.testing.assert_array_equal(model.po.ocv_interpolator.soc_grid.numpy(), expected_soc)
        np.testing.assert_array_equal(model.po.ocv_interpolator.v_grid.numpy(), expected_voltage)
    print("train-only OCV prior regression passed")


if __name__ == "__main__":
    main()
