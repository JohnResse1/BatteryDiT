"""Deterministic Transformer reconstruction baseline.

This is the formal Transformer comparison baseline.  It reconstructs voltage
from voltage and current without a mask channel, corruption curriculum, or
Physics Operator.
"""

import torch
import torch.nn as nn


class _TransformerBackbone(nn.Module):
    """Transformer encoder mapping normalized voltage/current to voltage."""

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_proj(inputs) + self.pos_emb[:, : inputs.size(1), :]
        return self.output_proj(self.encoder(encoded))


class Model(nn.Module):
    """Clean auto-encoding Transformer baseline with the shared recon API."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        seq_len = getattr(args, "seq_len", 1000)
        d_model = getattr(args, "d_model", 256)
        n_heads = getattr(args, "n_heads", 4)
        d_ff = getattr(args, "d_ff", 512)
        num_layers = getattr(args, "e_layers", 4)
        dropout = getattr(args, "dropout", 0.1)
        activation = getattr(args, "activation", "gelu")

        self.recon = _TransformerBackbone(
            seq_len=seq_len,
            in_channels=2,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )
        self.pred = nn.Identity()

        self.i_min = float(getattr(args, "i_min", 0.0))
        self.i_max = float(getattr(args, "i_max", 6.2))
        if self.i_max <= self.i_min:
            raise ValueError(f"i_max must be greater than i_min, got {self.i_min}, {self.i_max}")

    def _normalize_phy(self, voltage: torch.Tensor) -> torch.Tensor:
        return (voltage - self.args.v_min) / (self.args.v_max - self.args.v_min)

    def _denormalize_phy(self, voltage: torch.Tensor) -> torch.Tensor:
        return voltage * (self.args.v_max - self.args.v_min) + self.args.v_min

    def _normalize_current(self, current: torch.Tensor) -> torch.Tensor:
        return (current - self.i_min) / (self.i_max - self.i_min)

    def compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        return torch.mean((prediction - target) ** 2)

    def _prepare_inputs(self, voltage: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            torch.cat(
                [self._normalize_phy(voltage), self._normalize_current(current)],
                dim=-1,
            )
        )

    def forward(
        self,
        V: torch.Tensor,
        I: torch.Tensor,
        mode: str = "recon_only_train",
        steps: int = 1,
    ):
        if V.dim() == 2:
            V = V.unsqueeze(-1)
        if I.dim() == 2:
            I = I.unsqueeze(-1)

        prediction = self.recon(self._prepare_inputs(V, I))
        if mode == "recon_only_train":
            return prediction, self._normalize_phy(V), 0
        if mode == "recon_only_sampling":
            return prediction
        raise ValueError(f"Unsupported mode: {mode}")
