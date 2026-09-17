"""Formal dataset definitions and final supervised targets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    label: str
    nominal_capacity_ah: float | None = None

    def soh(self, capacity_ah: float, cell_nominal_capacity_ah: float | None = None) -> float:
        """Return the only SOH target written for this dataset."""
        if self.label == "second_life_capacity":
            return float(capacity_ah)
        nominal = self.nominal_capacity_ah or cell_nominal_capacity_ah
        if nominal is None or nominal <= 0:
            raise ValueError(f"{self.name} requires a positive nominal capacity")
        return float(capacity_ah) / float(nominal)


DATASETS = {
    "mit": DatasetProtocol("mit", "nominal_soh", nominal_capacity_ah=1.1),
    "hust": DatasetProtocol("hust", "nominal_soh", nominal_capacity_ah=1.1),
    "mich": DatasetProtocol("mich", "nominal_soh"),
    "sdu": DatasetProtocol("sdu", "second_life_capacity"),
    "stanford_2": DatasetProtocol("stanford_2", "nominal_soh", nominal_capacity_ah=0.24),
    "tju_nca_45c": DatasetProtocol("tju_nca_45c", "nominal_soh", nominal_capacity_ah=3.5),
    "tju_ncm_45c": DatasetProtocol("tju_ncm_45c", "nominal_soh", nominal_capacity_ah=3.5),
    "tju_ncm_nca_25c": DatasetProtocol("tju_ncm_nca_25c", "nominal_soh", nominal_capacity_ah=2.5),
}


def protocol(name: str) -> DatasetProtocol:
    try:
        return DATASETS[name]
    except KeyError as error:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown formal dataset {name!r}; choose from {choices}") from error
