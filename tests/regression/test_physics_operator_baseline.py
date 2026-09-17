"""Numerical and interface contract for the unified physics operator."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from models.physics_operator import OCVInterpolator, PhysicsOperator
from models.po_factory import create_physics_operator

SOC_GRID = [0.0, 0.5, 1.0]
VOLTAGE_GRID = [3.0, 3.5, 4.0]
CURRENT = torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[0.0], [2.0], [2.0], [4.0]]])
EXPECTED_SOC = torch.tensor([[0.0, 1.0 / 6.0, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0]])


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def main() -> None:
    interpolator = OCVInterpolator(SOC_GRID, VOLTAGE_GRID)
    assert_close(interpolator(torch.tensor([[-0.2, 0.25, 0.75, 1.2]])), torch.tensor([[2.8, 3.25, 3.75, 4.2]]))
    for profile in ("mit", "hust", "nca", "ncm", "ncm_nca"):
        operator = PhysicsOperator(profile, ocv_soc=SOC_GRID, ocv_voltage=VOLTAGE_GRID)
        output = operator(CURRENT)
        assert output.shape == CURRENT.shape
        assert_close(output[..., 0], 3.0 + EXPECTED_SOC)
        assert_close(operator(CURRENT[..., 0]), output)
        assert_close(operator(CURRENT, initial_soc=torch.tensor([[0.2], [0.4]]))[..., 0], torch.tensor([[3.2, 3.3333333, 3.6, 4.0], [3.4, 3.55, 3.7, 4.0]]))
        assert tuple(operator.state_dict()) == (
            "ocv_interpolator.soc_grid", "ocv_interpolator.v_grid",
            "ocv_interpolator.slope_low", "ocv_interpolator.slope_high",
        )
    expected_types = {"mit", "hust", "nca", "ncm", "ncm_nca", "mich", "sdu", "stanford_2"}
    for identifier in expected_types:
        operator = create_physics_operator(SimpleNamespace(po_type=identifier), ocv_soc=SOC_GRID, ocv_voltage=VOLTAGE_GRID)
        assert isinstance(operator, PhysicsOperator)
    print("unified physics-operator contract passed")


if __name__ == "__main__":
    main()
