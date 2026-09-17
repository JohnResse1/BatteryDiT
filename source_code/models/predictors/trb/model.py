import torch
import torch.nn as nn
from .blocks import (
    make_norm_1d, SqueezeExcitation1D,
    TemporalResidualBlock, AttentionPooling1D,
)


class Model(nn.Module):
    """
    TRB predictor: 1-channel input (V_recon or V_raw) → SOH, RUL.
    Same architecture as A1 / ReconOnlyPredictor.
    """

    def __init__(self, args=None, dropout=0.1, norm_type='gn', use_global_attn=True):
        super().__init__()
        self.args = args

        # Stem: 1-channel input
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, 1, 3, bias=False),
            make_norm_1d(64, norm_type), nn.GELU(),
            nn.Conv1d(64, 64, 5, 1, 2, bias=False),
            make_norm_1d(64, norm_type), nn.GELU(),
        )

        # 6 TRBs with increasing dilation
        block_cfgs = [
            (64, 64, 1, 1), (64, 96, 2, 1), (96, 128, 1, 2),
            (128, 160, 2, 2), (160, 192, 1, 4), (192, 192, 1, 4),
        ]
        self.backbone = nn.ModuleList()
        for c_in, c_out, s, d in block_cfgs:
            self.backbone.append(TemporalResidualBlock(c_in, c_out, s, d, dropout, norm_type))

        # Multi-pooling
        self.sequence_norm = nn.LayerNorm(192)
        if use_global_attn:
            self.global_attn = nn.MultiheadAttention(192, num_heads=4, batch_first=True, dropout=dropout)
        else:
            self.global_attn = None
        self.attn_pool = AttentionPooling1D(192)

        pool_dim = 192 * 4
        self.pool_gate = nn.Sequential(
            nn.Linear(pool_dim, max(pool_dim // 6, 128)), nn.GELU(),
            nn.Linear(max(pool_dim // 6, 128), 4), nn.Sigmoid(),
        )

        # Shared MLP
        self.shared = nn.Sequential(
            nn.Linear(pool_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        )

        # Heads
        self.q_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(64, 1),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(64, 1),
        )

    def encode(self, V):
        """Return the 128-dimensional degradation representation before the heads."""
        if V.dim() == 3:
            V = V.squeeze(-1)
        x = V.unsqueeze(1)                     # (B, 1, L)

        x = self.stem(x)                        # (B, 64, L)
        for trb in self.backbone:
            x = trb(x)                          # (B, 192, L/4)

        seq = x.transpose(1, 2)                 # (B, L/4, 192)
        seq = self.sequence_norm(seq)

        if self.global_attn is not None:
            seq, _ = self.global_attn(seq, seq, seq)

        avg = seq.mean(dim=1)                   # (B, 192)
        mx = seq.amax(dim=1)                    # (B, 192)
        std = torch.sqrt(torch.clamp(seq.var(dim=1, unbiased=False), min=1e-8))
        attn = self.attn_pool(seq)              # (B, 192)
        pools = torch.cat([avg, mx, std, attn], dim=-1)

        gates = self.pool_gate(pools)
        fused = (avg * gates[:, 0:1] + mx * gates[:, 1:2] +
                 std * gates[:, 2:3] + attn * gates[:, 3:4])

        features = torch.cat([fused] * 4, dim=-1)
        return self.shared(features)

    def forward(self, V, **kwargs):
        # V: (B, L) or (B, L, 1)
        h = self.encode(V)

        pred_q = self.q_head(h)
        pred_rul = self.rul_head(h)
        return torch.cat([pred_q, pred_rul], dim=-1)
