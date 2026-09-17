"""The fixed online corruption mixture shared by formal SOH predictors."""
from __future__ import annotations

import numpy as np
import torch


def corrupt_training_inputs(voltage: torch.Tensor, current: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply clean/downsample/segment/random/destructive draws per sample."""
    output_voltage, output_current = voltage.clone(), current.clone()
    boundaries = np.cumsum([float(args.p_clean), float(args.p_downsample), float(args.p_segment), float(args.p_random)])
    for index in range(voltage.size(0)):
        draw = float(torch.rand(()))
        v, i = voltage[index:index + 1], current[index:index + 1]
        if draw < boundaries[0]:
            continue
        if draw < boundaries[1]:
            mask = torch.zeros_like(v)
            mask[:, ::int(np.random.randint(2, 8))] = 1.0
            output_voltage[index:index + 1], output_current[index:index + 1] = v * mask, i * mask
        elif draw < boundaries[2]:
            length = max(1, int(v.size(1) * np.random.uniform(args.segment_ratio_min, args.segment_ratio_max)))
            start = int(np.random.randint(0, max(1, v.size(1) - length + 1)))
            mask = torch.ones_like(v)
            mask[:, start:start + length] = 0.0
            output_voltage[index:index + 1], output_current[index:index + 1] = v * mask, i * mask
        elif draw < boundaries[3]:
            mask = (torch.rand_like(v) > 0.15).float()
            output_voltage[index:index + 1] = v * mask + 0.03 * torch.randn_like(v) * (1.0 - mask)
            output_current[index:index + 1] = i * mask + 0.03 * torch.randn_like(i) * (1.0 - mask)
        else:
            length = max(1, int(v.size(1) * np.random.uniform(0.12, 0.25)))
            mask = torch.ones_like(v)
            for _ in range(2):
                start = int(np.random.randint(0, max(1, v.size(1) - length + 1)))
                mask[:, start:start + length] = 0.0
            output_voltage[index:index + 1] = v * mask + 0.05 * torch.randn_like(v) * (1.0 - mask)
            output_current[index:index + 1] = i * mask + 0.05 * torch.randn_like(i) * (1.0 - mask)
    return output_voltage, output_current
