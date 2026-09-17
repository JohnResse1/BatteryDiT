"""Build formal datasets from the raw sources stored in this repository."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .datasets import DATASETS
from .sources import datasets, mit

# Set up paths
SOURCE_CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = SOURCE_CODE_ROOT / "data"
DEFAULT_SPLIT_ROOT = Path(__file__).resolve().parent / "splits"
RAW_SOURCE_ROOT = SOURCE_CODE_ROOT / "data" / "raw_sources"
DEFAULT_MIT_SOURCE = RAW_SOURCE_ROOT / "mit"
DEFAULT_DATASET_ROOT = RAW_SOURCE_ROOT / "datasets"


# Parse command-line arguments required to build datasets.
# Custom dataset can be added to the DATASETS dictionary in datasets.py and then specified with --dataset.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*sorted(DATASETS), "all"), required=True) # Add your dataset names here
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mit-source", type=Path, default=DEFAULT_MIT_SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--replace", action="store_true", help="replace an existing completed dataset after a successful staged build")
    return parser.parse_args()

# Split of the dataset is set for reproducibility.
# Custom splits can be defined in the splits directory as a JSON file with train, vali, and test keys.
def _splits(dataset: str) -> dict[str, list[str]]:
    path = DEFAULT_SPLIT_ROOT / f"{dataset}.json"
    with path.open() as handle:
        values = json.load(handle)
    if set(values) != {"train", "vali", "test"}:
        raise ValueError(f"{path} must define train, vali, and test splits")
    return values


def _build(dataset: str, staging_root: Path, options: argparse.Namespace) -> int:
    splits = _splits(dataset)
    if dataset == "mit":
        return mit.build(staging_root, options.mit_source.resolve(), splits)
    return datasets.build(dataset, staging_root, options.dataset_root.resolve(), splits)


# Publish the dataset by building it in a staging directory and then moving it to the final destination.
# We publish the dataset to /data/datasets/<dataset> and the staging directory is /data/datasets/.<dataset>.building.
def _publish(dataset: str, output_root: Path, options: argparse.Namespace) -> int:
    destination = output_root / dataset
    staging = output_root / f".{dataset}.building"
    if staging.exists():
        raise FileExistsError(f"remove interrupted staging directory first: {staging}")
    if destination.exists() and not options.replace:
        raise FileExistsError(f"{destination} exists; use --replace only after validating a rebuild")
    staging.mkdir(parents=True)
    try:
        count = _build(dataset, staging, options)
        built = staging / dataset
        if not built.is_dir() or not any(built.rglob("*.npz")):
            raise RuntimeError(f"{dataset} build produced no samples")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(built, destination)
        staging.rmdir()
        return count
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    options = parse_args()
    output_root = options.output_root.resolve()
    selected = sorted(DATASETS) if options.dataset == "all" else [options.dataset]
    for dataset in selected:
        count = _publish(dataset, output_root, options)
        print(f"Built {count:,} {dataset} samples under {output_root / dataset}")


if __name__ == "__main__":
    main()
