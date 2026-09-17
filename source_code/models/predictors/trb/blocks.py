import torch
import torch.nn as nn


def make_norm_1d(channels, norm_type='gn', groups=16):
    norm = (norm_type or 'gn').lower()
    if norm == 'bn':
        return nn.BatchNorm1d(channels)
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=max(g, 1), num_channels=channels)


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class TemporalResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, dilation=1, dropout=0.1, norm_type='gn'):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride, padding, dilation, bias=False)
        self.norm1 = make_norm_1d(c_out, norm_type)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, 1, padding, dilation, bias=False)
        self.norm2 = make_norm_1d(c_out, norm_type)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.se = SqueezeExcitation1D(c_out)
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                make_norm_1d(c_out, norm_type),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        r = self.shortcut(x)
        o = self.act(self.norm1(self.conv1(x)))
        o = self.drop(o)
        o = self.conv2(o)
        o = self.norm2(o)
        o = self.se(o)
        return self.act(o + r)


class AttentionPooling1D(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        h = hidden_dim or max(dim // 2, 32)
        self.score = nn.Sequential(nn.Linear(dim, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(dim=1)
