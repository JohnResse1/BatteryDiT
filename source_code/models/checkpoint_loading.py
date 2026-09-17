"""Compatibility loading for reconstruction-only checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


LEGACY_DOWNSTREAM_PREFIX = "pred."


def state_dict_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the state dict from a common checkpoint payload."""
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def reconstruction_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Remove the historical, unused downstream feature-MLP parameters."""
    return OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not key.startswith(LEGACY_DOWNSTREAM_PREFIX)
    )


def load_reconstruction_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
    assign: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load all reconstruction parameters strictly after filtering ``pred.*``."""
    filtered = reconstruction_state_dict(state_dict)
    # Historical checkpoints may contain a 1000-point OCV table while a new
    # model was instantiated with another explicit table.  OCV buffers are
    # deterministic data, not learned weights; resize them before strict load.
    interpolator = getattr(getattr(model, "po", None), "ocv_interpolator", None)
    if interpolator is not None:
        for name in ("soc_grid", "v_grid", "slope_low", "slope_high"):
            key = f"po.ocv_interpolator.{name}"
            if key in filtered:
                interpolator._buffers[name] = filtered[key].detach().clone()
    try:
        return nn.Module.load_state_dict(model, filtered, strict=strict, assign=assign)
    except TypeError:  # Supports older PyTorch releases without ``assign``.
        return nn.Module.load_state_dict(model, filtered, strict=strict)


def load_reconstruction_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    assign: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load a historical local checkpoint into a reconstruction-only model."""
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping checkpoint payload, got {type(payload)!r}")
    return load_reconstruction_state_dict(
        model,
        state_dict_from_checkpoint(payload),
        strict=strict,
        assign=assign,
    )
