"""Regression contract for loading historical reconstruction checkpoints."""

from __future__ import annotations

import torch
from torch import nn

from models.checkpoint_loading import (
    load_reconstruction_state_dict,
    state_dict_from_checkpoint,
)


def main() -> None:
    model = nn.Linear(2, 1, bias=False)
    legacy_state = {
        "weight": torch.tensor([[2.0, -3.0]]),
        "pred.head.weight": torch.ones(1, 1),
        "pred.head.bias": torch.zeros(1),
    }
    result = load_reconstruction_state_dict(model, legacy_state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    torch.testing.assert_close(model.weight, legacy_state["weight"])
    # New reconstruction runs wrap their state under ``model``; legacy runs
    # used either ``model_state_dict`` or ``state_dict``.
    for envelope in ("model", "model_state_dict", "state_dict"):
        assert state_dict_from_checkpoint({envelope: legacy_state}) is legacy_state
    print("reconstruction checkpoint-loading regression passed")


if __name__ == "__main__":
    main()
