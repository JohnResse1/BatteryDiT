"""Physics-operator registrations for the retained formal datasets."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType

@dataclass(frozen=True)
class DatasetPhysics:
    po_type: str
    v_min: float
    v_max: float

DATASET_PHYSICS = MappingProxyType({
    # Locked benchmark-819 normalization for the MIT/LFP experiment.
    "mit": DatasetPhysics("MIT", 2.0, 3.6),
    "hust": DatasetPhysics("HUST", 1.5, 4.2),
    "mich": DatasetPhysics("NCM", 3.0, 4.2),
    "sdu": DatasetPhysics("NCM", 2.5, 4.2),
    "stanford_2": DatasetPhysics("NCM", 3.0, 4.4),
    "tju_nca_45c": DatasetPhysics("NCA", 2.5, 4.2),
    "tju_ncm_45c": DatasetPhysics("NCM", 2.5, 4.2),
    "tju_ncm_nca_25c": DatasetPhysics("NCM_NCA", 2.5, 4.2),
})

def normalize_physics_identifier(identifier: object) -> str:
    return str(identifier).strip().lower().replace("-", "_").replace(" ", "_")

def dataset_physics(dataset: object) -> DatasetPhysics:
    normalized = normalize_physics_identifier(dataset)
    try: return DATASET_PHYSICS[normalized]
    except KeyError as error: raise KeyError(f"No physics registration for {dataset!r}") from error

def resolve_po_type(identifier: object) -> str:
    normalized = normalize_physics_identifier(identifier)
    spec = DATASET_PHYSICS.get(normalized)
    return spec.po_type.lower() if spec is not None else normalized
