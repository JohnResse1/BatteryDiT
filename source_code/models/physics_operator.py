"""Unified chemistry-aware physics operator for BatteryDiT."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    current_limit: float
    fallback_soc: tuple[float, ...]
    fallback_voltage: tuple[float, ...]


_PROFILES = {
    # These curves are retained only as deterministic legacy fallbacks.
    # New dataset runs provide a fixed training-split OCV table.
    "mit": PhysicsProfile("MIT", 20.0,
        (-.05, 0., .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99, 1., 1.05),
        (1.5, 2., 2.8, 3.05, 3.15, 3.2, 3.23, 3.25, 3.27, 3.285, 3.3, 3.32, 3.34, 3.38, 3.45, 3.55, 3.6, 4.2)),
    "hust": PhysicsProfile("HUST", 10.0,
        (-.05, 0., .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99, 1., 1.05),
        (1.5, 2., 2.8, 3.05, 3.15, 3.2, 3.23, 3.25, 3.27, 3.285, 3.3, 3.32, 3.34, 3.38, 3.45, 3.55, 3.6, 4.2)),
    "nca": PhysicsProfile("NCA", 20.0,
        (-.05, 0., .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1., 1.05),
        (2.5, 2.65, 3.25, 3.42, 3.53, 3.62, 3.7, 3.79, 3.88, 3.97, 4.05, 4.13, 4.17, 4.2, 4.3)),
    "ncm": PhysicsProfile("NCM", 20.0,
        (-.05, 0., .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1., 1.05),
        (2.4, 2.5, 3.3, 3.48, 3.54, 3.58, 3.62, 3.65, 3.68, 3.71, 3.74, 3.77, 3.8, 3.84, 3.88, 3.92, 3.96, 4.01, 4.06, 4.1, 4.15, 4.2, 4.3)),
    "ncm_nca": PhysicsProfile("NCM_NCA", 20.0,
        (-.05, 0., .01, .03, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .98, 1., 1.05),
        (2.4, 2.5, 3., 3.3, 3.42, 3.52, 3.58, 3.63, 3.665, 3.7, 3.73, 3.76, 3.79, 3.82, 3.85, 3.89, 3.93, 3.97, 4.02, 4.07, 4.11, 4.15, 4.18, 4.192, 4.2, 4.3)),
}


def physics_profile(name: str) -> PhysicsProfile:
    try:
        return _PROFILES[name.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unknown physics profile: {name!r}") from error


class OCVInterpolator(nn.Module):
    """Differentiable piecewise-linear SOC--OCV interpolation with extrapolation."""

    def __init__(self, soc_points, voltage_points):
        super().__init__()
        soc = torch.as_tensor(soc_points, dtype=torch.float32).flatten()
        voltage = torch.as_tensor(voltage_points, dtype=torch.float32).flatten()
        if soc.numel() < 2 or soc.numel() != voltage.numel():
            raise ValueError("OCV tables require at least two matching SOC and voltage points")
        if not bool(torch.all(soc[1:] > soc[:-1])):
            raise ValueError("SOC grid must be strictly increasing")
        self.register_buffer("soc_grid", soc)
        self.register_buffer("v_grid", voltage)
        # Keep the historical ``+ 1e-6`` denominator: old checkpoints and
        # regression outputs used this exact boundary-extrapolation rule.
        self.register_buffer("slope_low", (voltage[1] - voltage[0]) / (soc[1] - soc[0] + 1e-6))
        self.register_buffer("slope_high", (voltage[-1] - voltage[-2]) / (soc[-1] - soc[-2] + 1e-6))

    def forward(self, query_soc: torch.Tensor) -> torch.Tensor:
        bounded = query_soc.clamp(self.soc_grid[0], self.soc_grid[-1])
        indices = torch.searchsorted(self.soc_grid.detach(), bounded, right=True)
        indices = indices.clamp(1, self.soc_grid.numel() - 1)
        low, high = indices - 1, indices
        soc_low, soc_high = self.soc_grid[low], self.soc_grid[high]
        voltage_low, voltage_high = self.v_grid[low], self.v_grid[high]
        weight = (bounded - soc_low) / (soc_high - soc_low).clamp_min(1e-6)
        interpolated = voltage_low + weight * (voltage_high - voltage_low)
        return (
            interpolated
            + torch.relu(query_soc - self.soc_grid[-1]) * self.slope_high
            - torch.relu(self.soc_grid[0] - query_soc) * self.slope_low
        )


class PhysicsOperator(nn.Module):
    """Unified relative-charge SOC--OCV prior with one stable input/output contract.

    Inputs are current sequences shaped ``(B, L)`` or ``(B, L, 1)``.  Outputs
    are always voltage sequences shaped ``(B, L, 1)``.  Chemistry affects only
    the named profile and its current safety limit; all SOC and OCV calculations
    follow this one implementation.
    """

    def __init__(self, profile: str, current_scale: float = 1.0, ocv_soc=None, ocv_voltage=None):
        super().__init__()
        self.profile = physics_profile(profile)
        self.current_scale = float(current_scale)
        if ocv_soc is None or ocv_voltage is None:
            ocv_soc = np.asarray(self.profile.fallback_soc, dtype=np.float32)
            ocv_voltage = np.asarray(self.profile.fallback_voltage, dtype=np.float32)
        # The child name is a historical checkpoint contract.
        self.ocv_interpolator = OCVInterpolator(ocv_soc, ocv_voltage)

    def forward(self, current: torch.Tensor, initial_soc=None) -> torch.Tensor:
        if current.dim() == 2:
            current = current.unsqueeze(-1)
        if current.dim() != 3 or current.size(-1) != 1:
            raise ValueError(f"current must have shape (B, L) or (B, L, 1), got {tuple(current.shape)}")
        batch_size, length, _ = current.shape
        safe_current = (current * self.current_scale).clamp(
            min=-self.profile.current_limit, max=self.profile.current_limit
        )
        integrated = torch.cumsum(safe_current, dim=1)
        total = integrated[:, -1:, :].clamp_min(1e-6)
        progress = (integrated / total).reshape(batch_size, length)
        if initial_soc is None:
            start = torch.zeros(batch_size, 1, device=current.device, dtype=current.dtype)
        elif isinstance(initial_soc, (float, int)):
            start = torch.full((batch_size, 1), float(initial_soc), device=current.device, dtype=current.dtype)
        else:
            start = torch.as_tensor(initial_soc, device=current.device, dtype=current.dtype).reshape(batch_size, 1)
        soc = start + progress * (1.0 - start)
        return self.ocv_interpolator(soc).clamp(0.0, 5.0).unsqueeze(-1)
