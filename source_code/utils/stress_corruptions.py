"""Frozen corruption protocol used only by evaluation stress tests.

Training keeps its stochastic ``generate_hybrid_mask`` augmentation.  Stress
tests instead need each nominal severity to correspond to an exact number of
missing time points in every sample.  Reconstruction and downstream stress
tests import the same constants and reset the RNG per condition/level so that
they evaluate the same corruptions regardless of execution order.
"""

import random

import numpy as np
import torch


FORMAL_STRESS_PROTOCOL = "batterydit_exact_10_30_50_70_90_v3"
FORMAL_MISSING_RATES = (0.10, 0.30, 0.50, 0.70, 0.90)
FORMAL_DOWNSAMPLE_KEEP_RATIOS = (0.90, 0.70, 0.50, 0.30, 0.10)
FORMAL_DESTRUCTIVE_NOISE_LEVELS = (0.01, 0.03, 0.05, 0.10, 0.15)
FORMAL_HYBRID_BLOCK_FRACTION = 0.50
STRESS_SEED_SCHEME = "base_seed_plus_condition_offset_plus_level_v1"
STRESS_CONDITIONS = (
    "downsample", "random_missing", "segment_missing", "destructive", "hybrid",
)

_CONDITION_SEED_OFFSETS = {
    "downsample": 10_000,
    "destructive": 20_000,
    "random_missing": 30_000,
    "segment_missing": 40_000,
    "hybrid": 50_000,
}


def reset_stress_rng(base_seed, condition, level_index):
    """Reset every RNG to a stable condition/level-specific seed."""
    if condition not in _CONDITION_SEED_OFFSETS:
        raise ValueError(
            f"Unknown stress condition {condition!r}; "
            f"choose from {sorted(_CONDITION_SEED_OFFSETS)}"
        )
    if not 0 <= int(level_index) < len(FORMAL_MISSING_RATES):
        raise ValueError(f"Invalid stress level index: {level_index}")
    seed = int(base_seed) + _CONDITION_SEED_OFFSETS[condition] + int(level_index)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _missing_count(seq_len, missing_rate):
    if not 0.0 <= float(missing_rate) <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
    return min(seq_len, max(0, int(round(seq_len * float(missing_rate)))))


def exact_downsample_mask(batch_size, seq_len, keep_ratio, device):
    """Keep an exact uniform grid with a seeded phase for every sample."""
    if not 0.0 <= float(keep_ratio) <= 1.0:
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
    keep_count = min(seq_len, max(1, int(round(seq_len * float(keep_ratio)))))
    base_indices = torch.linspace(
        0, seq_len - 1, steps=keep_count, device=device
    ).round().long()
    phases = torch.randint(0, seq_len, (batch_size,), device=device)
    mask = torch.zeros(batch_size, seq_len, 1, device=device)
    for batch_index, phase in enumerate(phases.tolist()):
        indices = (base_indices + phase) % seq_len
        mask[batch_index, indices, 0] = 1.0
    return mask


def exact_random_missing_mask(batch_size, seq_len, missing_rate, device):
    """Mask exactly ``round(seq_len * missing_rate)`` random points/sample."""
    count = _missing_count(seq_len, missing_rate)
    mask = torch.ones(batch_size, seq_len, 1, device=device)
    if count == 0:
        return mask
    for batch_index in range(batch_size):
        indices = torch.randperm(seq_len, device=device)[:count]
        mask[batch_index, indices, 0] = 0.0
    return mask


def exact_segment_missing_mask(batch_size, seq_len, missing_rate, device):
    """Mask one contiguous block with an exact length in every sample."""
    count = _missing_count(seq_len, missing_rate)
    mask = torch.ones(batch_size, seq_len, 1, device=device)
    if count == 0:
        return mask
    max_start = seq_len - count + 1
    starts = torch.randint(0, max_start, (batch_size,), device=device)
    for batch_index, start in enumerate(starts.tolist()):
        mask[batch_index, start : start + count, 0] = 0.0
    return mask


def exact_hybrid_missing_mask(
    batch_size,
    seq_len,
    missing_rate,
    device,
    block_fraction=0.5,
):
    """Mask an exact total: one block plus random points outside the block.

    Half of the requested missing points form a contiguous block by default;
    the remainder are sampled uniformly from the still-observed positions.
    """
    if not 0.0 <= float(block_fraction) <= 1.0:
        raise ValueError(
            f"block_fraction must be in [0, 1], got {block_fraction}"
        )
    total_count = _missing_count(seq_len, missing_rate)
    block_count = int(round(total_count * float(block_fraction)))
    random_count = total_count - block_count
    mask = torch.ones(batch_size, seq_len, 1, device=device)
    if total_count == 0:
        return mask

    max_start = seq_len - block_count + 1
    starts = torch.randint(0, max_start, (batch_size,), device=device)
    for batch_index, start in enumerate(starts.tolist()):
        if block_count:
            mask[batch_index, start : start + block_count, 0] = 0.0
        if random_count:
            candidates = torch.nonzero(
                mask[batch_index, :, 0] > 0.5, as_tuple=False
            ).flatten()
            selected = candidates[
                torch.randperm(candidates.numel(), device=device)[:random_count]
            ]
            mask[batch_index, selected, 0] = 0.0
    return mask


def apply_stress_corruption(condition, level_index, voltage, current):
    """Apply one formal L1–L5 corruption to a clean voltage/current batch."""
    level_index = int(level_index)
    if condition not in STRESS_CONDITIONS:
        raise ValueError(f"unknown stress condition {condition!r}; choose from {STRESS_CONDITIONS}")
    if not 0 <= level_index < len(FORMAL_MISSING_RATES):
        raise ValueError(f"invalid stress level index: {level_index}")
    batch_size, sequence_length, _ = voltage.shape
    missing_rate = FORMAL_MISSING_RATES[level_index]
    noise_level = FORMAL_DESTRUCTIVE_NOISE_LEVELS[level_index]
    if condition == "downsample":
        mask = exact_downsample_mask(
            batch_size, sequence_length,
            FORMAL_DOWNSAMPLE_KEEP_RATIOS[level_index], voltage.device,
        )
        return voltage * mask, current * mask
    if condition == "random_missing":
        mask = exact_random_missing_mask(batch_size, sequence_length, missing_rate, voltage.device)
        return voltage * mask, current * mask
    if condition == "segment_missing":
        mask = exact_segment_missing_mask(batch_size, sequence_length, missing_rate, voltage.device)
        return voltage * mask, current * mask
    from .tools import inject_compound_noise_torch
    if condition == "destructive":
        return inject_compound_noise_torch(voltage, current, noise_level)
    mask = exact_hybrid_missing_mask(
        batch_size, sequence_length, missing_rate, voltage.device,
        FORMAL_HYBRID_BLOCK_FRACTION,
    )
    noisy_voltage, noisy_current = inject_compound_noise_torch(voltage, current, noise_level)
    return noisy_voltage * mask, noisy_current * mask
