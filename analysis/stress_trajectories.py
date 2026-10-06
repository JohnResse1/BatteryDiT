"""Export one formal MIT stress-test trajectory for Fig. S5.

The formal stress experiment stores aggregate L1--L5 metrics, while the
supplementary trajectory panel needs one clean/corrupted/reconstructed curve
for every condition and severity.  This module runs the frozen BatteryDiT
checkpoint with the same corruption protocol and evaluation seed as
``analysis.stress_test`` and writes only figure-ready arrays to
``analysis/outputs/``.

No plotting is performed here.  The representative cycle is selected once by
the median-error rule used by the legacy SI panel: average the L3 waveform
RMSE across downsampling, segment missingness, and destructive noise, then
choose the cycle nearest the pooled median.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analysis.utils.io import analysis_dir
from analysis.utils.paths import REPOSITORY_ROOT


SOURCE_CODE = REPOSITORY_ROOT / "source_code"
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))

from data_provider.data_loader import BatteryDataset  # noqa: E402
from exp.reconstruction_stress_test import Exp_Reconstruction_Stress_Test  # noqa: E402
from main import configure_dataset, configure_model_protocol, parser  # noqa: E402
from utils.stress_corruptions import (  # noqa: E402
    FORMAL_STRESS_PROTOCOL,
    apply_stress_corruption,
    reset_stress_rng,
)


DATASET = "mit"
BASE_SEED = 3407
MODEL_SEED = 2021
BATCH_SIZE = 512
SELECTION_LEVEL_INDEX = 2  # L3
SELECTION_CONDITIONS = ("downsample", "segment_missing", "destructive")
CONDITIONS = (
    "downsample",
    "random_missing",
    "segment_missing",
    "destructive",
    "hybrid",
)
LEVELS = (1, 2, 3, 4, 5)


def _runner() -> Exp_Reconstruction_Stress_Test:
    """Build the formal BatteryDiT runner around the frozen checkpoint."""
    args = parser().parse_args([
        "--task_name", "ReconstructionStressTest",
        "--is_training", "0",
        "--model_id", f"seed{MODEL_SEED}",
        "--recon_model", "batterydit",
        "--data", DATASET,
        "--seed", str(MODEL_SEED),
        "--stress_seed", str(BASE_SEED),
        "--eval_batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--use_gpu", "1",
        "--gpu", "0",
    ])
    configure_dataset(args)
    configure_model_protocol(args)
    args.ocv_table_path = str(SOURCE_CODE / "models" / "ocv" / "mit.npz")
    checkpoint = (
        SOURCE_CODE / "checkpoints" / "Reconstruction" / DATASET /
        f"Reconstruction_seed{MODEL_SEED}_batterydit_{DATASET}_0" /
        "checkpoint.pth"
    )
    args.recn_chkpt = str(checkpoint)
    runner = Exp_Reconstruction_Stress_Test(args)
    runner._load_best_checkpoint(checkpoint)
    runner.model.eval()
    return runner


def _batch_prediction(runner, condition: str, level_index: int, batch):
    current, voltage, _, _ = batch
    current = current.to(runner.device, non_blocking=True)
    voltage = voltage.to(runner.device, non_blocking=True)
    corrupted_voltage, corrupted_current = apply_stress_corruption(
        condition, level_index, voltage, current
    )
    reconstructed = runner._predict_physical(corrupted_voltage, corrupted_current)
    rmse = torch.sqrt(torch.mean((reconstructed - voltage) ** 2, dim=(1, 2)))
    return (
        corrupted_voltage.detach().cpu().numpy().reshape(len(voltage), -1),
        reconstructed.detach().cpu().numpy().reshape(len(voltage), -1),
        rmse.detach().cpu().numpy(),
    )


@torch.inference_mode()
def _selection_scores(runner, dataset: BatteryDataset) -> np.ndarray:
    """Compute the locked representative score on the complete test split."""
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    scores = []
    for condition in SELECTION_CONDITIONS:
        reset_stress_rng(BASE_SEED, condition, SELECTION_LEVEL_INDEX)
        values = []
        for batch in loader:
            _, _, rmse = _batch_prediction(
                runner, condition, SELECTION_LEVEL_INDEX, batch
            )
            values.append(rmse)
        scores.append(np.concatenate(values))
    return np.mean(np.stack(scores), axis=0)


@torch.inference_mode()
def _selected_curves(
    runner,
    dataset: BatteryDataset,
    selected_index: int,
) -> tuple[np.ndarray, dict[tuple[str, int], dict[str, np.ndarray | float]]]:
    """Evaluate the selected cycle while preserving formal batch boundaries."""
    prefix_length = min(
        ((selected_index // BATCH_SIZE) + 1) * BATCH_SIZE,
        len(dataset),
    )
    loader = DataLoader(
        Subset(dataset, range(prefix_length)), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    selected_record = dataset[selected_index]
    clean = selected_record[1].numpy().reshape(-1).astype(np.float32)
    curves: dict[tuple[str, int], dict[str, np.ndarray | float]] = {}

    for condition in CONDITIONS:
        for level in LEVELS:
            level_index = level - 1
            reset_stress_rng(BASE_SEED, condition, level_index)
            offset = 0
            found = None
            for batch in loader:
                corrupted, reconstructed, rmse = _batch_prediction(
                    runner, condition, level_index, batch
                )
                batch_stop = offset + len(rmse)
                if offset <= selected_index < batch_stop:
                    local = selected_index - offset
                    found = {
                        "corrupted": corrupted[local].astype(np.float32),
                        "reconstructed": reconstructed[local].astype(np.float32),
                        "rmse": float(rmse[local]),
                    }
                    break
                offset = batch_stop
            if found is None:
                raise RuntimeError(
                    f"selected index {selected_index} was not reached for "
                    f"{condition} L{level}"
                )
            curves[(condition, level)] = found
    return clean, curves


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export formal Fig. S5 trajectories")

    dataset = BatteryDataset(DATASET, split="test", condition="clean")
    runner = _runner()
    scores = _selection_scores(runner, dataset)
    median_score = float(np.median(scores))
    selected_index = int(np.argmin(np.abs(scores - median_score)))
    clean, curves = _selected_curves(runner, dataset, selected_index)
    selected_key = str(dataset.paths[selected_index].relative_to(dataset.root))

    arrays: dict[str, np.ndarray] = {
        "levels": np.asarray(LEVELS, dtype=np.int64),
        "topologies": np.asarray(CONDITIONS),
        "clean": clean,
    }
    metadata_conditions = {}
    for (condition, level), item in curves.items():
        key = f"{condition}_L{level}"
        arrays[f"{key}_corrupted"] = np.asarray(item["corrupted"])
        arrays[f"{key}_reconstructed"] = np.asarray(item["reconstructed"])
        arrays[f"{key}_rmse"] = np.asarray(float(item["rmse"]), dtype=np.float64)
        metadata_conditions[key] = {"waveform_rmse_v": float(item["rmse"])}

    output = analysis_dir()
    np.savez_compressed(output / "stress_test_representative_trajectories.npz", **arrays)
    metadata = {
        "dataset": DATASET,
        "protocol": FORMAL_STRESS_PROTOCOL,
        "evaluation_seed": BASE_SEED,
        "model_seed": MODEL_SEED,
        "model": "BatteryDiT",
        "checkpoint": str((SOURCE_CODE / "checkpoints" / "Reconstruction" /
                            DATASET /
                            f"Reconstruction_seed{MODEL_SEED}_batterydit_{DATASET}_0" /
                            "checkpoint.pth").relative_to(REPOSITORY_ROOT)),
        "selection_rule": (
            "MIT clean test cycle nearest the pooled median of its mean "
            "BatteryDiT waveform RMSE across Downsample/Segment-missing/"
            "Destructive L3"
        ),
        "selection_conditions": list(SELECTION_CONDITIONS),
        "selection_level": "L3",
        "selected_index": selected_index,
        "selected_key": selected_key,
        "selection_score": float(scores[selected_index]),
        "selection_median": median_score,
        "conditions": metadata_conditions,
    }
    (output / "stress_test_representative_trajectories.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
