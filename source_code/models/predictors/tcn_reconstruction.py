"""TCN downstream baseline on one reconstructed-voltage channel."""
from __future__ import annotations

from torch import nn

from .direct import _TCNBlock


class Model(nn.Module):
    """The formal BatteryDiT→TCN comparator."""

    def __init__(self, args=None, dropout: float = 0.1):
        super().__init__()
        channels, dilations = [1, 64, 64, 128, 128], [1, 2, 4, 8]
        self.blocks = nn.Sequential(*[
            _TCNBlock(channels[index], channels[index + 1], dilations[index], dropout)
            for index in range(len(channels) - 1)
        ])
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, voltage, **kwargs):
        if voltage.dim() == 3:
            voltage = voltage.squeeze(-1)
        return self.head(self.blocks(voltage.unsqueeze(1)).mean(dim=-1))
