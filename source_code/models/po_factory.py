"""Construction of the unified chemistry-aware physics operator."""
from __future__ import annotations

from torch import nn

from .dataset_physics import resolve_po_type
from .physics_operator import PhysicsOperator


def create_physics_operator(
    args,
    *,
    current_scale: float = 1.0,
    ocv_soc=None,
    ocv_voltage=None,
    prefix: str = "[Physics Operator]",
) -> nn.Module:
    """Create the one PO implementation using a canonical chemistry profile."""
    profile = resolve_po_type(getattr(args, "po_type", "MIT"))
    if profile not in {"mit", "hust", "nca", "ncm", "ncm_nca"}:
        raise ValueError(f"{prefix} unknown po_type={profile!r}")
    print(f"{prefix} Using {profile.upper()} physics profile")
    return PhysicsOperator(
        profile, current_scale=current_scale, ocv_soc=ocv_soc, ocv_voltage=ocv_voltage
    )
