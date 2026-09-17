"""Construct DataLoaders for one formal dataset at a time."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data_loader import DATA_ROOT, BatteryDataset


def data_provider(args, flag: str, error_type: str | None = None):
    dataset = BatteryDataset(
        args.data,
        split=flag,
        condition=error_type,
        data_root=getattr(args, "data_root", None) or DATA_ROOT,
    )
    is_train = flag == "train"
    batch_size = args.batch_size if is_train else (getattr(args, "eval_batch_size", None) or args.batch_size * 8)
    # Keep shuffling independent of model initialization and augmentation RNG.
    # The same formal seed therefore produces the same sample order for every
    # reconstruction model and for the downstream TRB stage.
    generator = torch.Generator().manual_seed(int(getattr(args, "seed", 2021))) if is_train else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    print(f"{dataset.dataset}/{dataset.split}/{dataset.condition}: {len(dataset):,} samples")
    return dataset, loader


def paired_data_provider(args, flag: str, error_type: str):
    """Return corrupted inputs paired to clean targets by formal sample key."""
    from .data_loader import PairedBatteryDataset

    dataset = PairedBatteryDataset(
        args.data,
        split=flag,
        condition=error_type,
        data_root=getattr(args, "data_root", None) or DATA_ROOT,
    )
    batch_size = getattr(args, "eval_batch_size", None) or args.batch_size * 8
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"{dataset.dataset}/{dataset.split}/{dataset.condition}: {len(dataset):,} paired samples")
    return dataset, loader
