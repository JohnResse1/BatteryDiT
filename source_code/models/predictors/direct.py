"""Direct SOH/RUL baselines used by the formal benchmark."""
from __future__ import annotations

import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden, layers = int(getattr(args, "lstm_hidden", 128)), int(getattr(args, "lstm_layers", 2))
        dropout = float(getattr(args, "dropout", 0.1))
        self.lstm = nn.LSTM(2, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(values)
        return self.head(hidden[-1])


class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden, layers = int(getattr(args, "gru_hidden", 128)), int(getattr(args, "gru_layers", 2))
        dropout = float(getattr(args, "dropout", 0.1))
        self.gru = nn.GRU(2, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(values)
        return self.head(hidden[-1])


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        width = int(getattr(args, "tf_d_model", 128))
        heads, layers = int(getattr(args, "tf_nhead", 4)), int(getattr(args, "tf_layers", 2))
        dropout = float(getattr(args, "dropout", 0.1))
        self.input_proj = nn.Linear(2, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, width))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_proj(values) + self.pos_embed[:, : values.size(1)])
        return self.head(self.norm(encoded).mean(dim=1))


class _TCNBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(min(channels_out, 16), channels_out), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels_out, channels_out, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(min(channels_out, 16), channels_out), nn.GELU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(values) + self.skip(values)


class TCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        channels = list(getattr(args, "tcn_channels", [2, 64, 64, 128, 128]))
        dilations = list(getattr(args, "tcn_dilations", [1, 2, 4, 8]))
        dropout = float(getattr(args, "dropout", 0.1))
        self.blocks = nn.Sequential(*[
            _TCNBlock(channels[index], channels[index + 1], dilations[index % len(dilations)], dropout)
            for index in range(len(channels) - 1)
        ])
        self.head = nn.Sequential(nn.Linear(channels[-1], 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(values.permute(0, 2, 1)).mean(dim=-1))
