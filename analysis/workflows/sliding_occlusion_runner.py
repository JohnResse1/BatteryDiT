"""Sliding-window occlusion and the support-vulnerability relationship.

This is the only mechanism workflow that needs BatteryDiT forward passes: each
occlusion window is reconstructed with the FROZEN formal checkpoints.  It is
submitted to Slurm.

Everything else is reused unchanged: the certified decomposition
(`exp/population_individual.py`), the formal metric definitions
(`evaluate_reconstruction`), the model loader and forward path of the new
repository, and the checkpoint layout under `source_code/checkpoints/`.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from analysis.utils.paths import REPOSITORY_ROOT

sys.path.insert(0, str(REPOSITORY_ROOT / "source_code"))

from data_provider.data_loader import BatteryDataset        # noqa: E402
from exp.exp_basic import Exp_Basic                          # noqa: E402
from exp.reconstruction_stress_test import Exp_Reconstruction_Stress_Test  # noqa: E402

from . import _certified as C
from .spatial_support import window_definitions

WIDTHS = (0.10, 0.20, 0.30)
DEFAULT_SEEDS = (2021, 2022, 2023)
DATASET = "mit"


def _reconstruct(runner, current: np.ndarray, voltage: np.ndarray,
                 mask: np.ndarray, device: str) -> np.ndarray:
    """Frozen one-step reconstruction of a masked cycle (V and I share the mask).

    Uses the formal runner's own `_predict_physical`, so the output is the
    denormalised physical voltage that the manuscript metrics are computed on.
    """
    corrupted_v = torch.tensor(voltage * mask, device=device,
                               dtype=torch.float32).reshape(1, -1, 1)
    corrupted_i = torch.tensor(current * mask, device=device,
                               dtype=torch.float32).reshape(1, -1, 1)
    with torch.no_grad():
        prediction = runner._predict_physical(corrupted_v, corrupted_i)
    return prediction.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _load_cycles(dataset, length: int):
    """All clean test cycles as (voltage, current) matrices on the fixed grid."""
    voltage, current, cells, cycles = [], [], [], []
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as sample:
            voltage.append(np.asarray(sample["V"], dtype=np.float32).reshape(-1))
            current.append(np.asarray(sample["I"], dtype=np.float32).reshape(-1))
        key = str(path.relative_to(dataset.root))
        cells.append(key.rsplit("/", 1)[0])
        cycles.append(int(key.rsplit("/", 1)[1].removesuffix(".npz")))
    return (np.stack(voltage), np.stack(current), np.asarray(cells), np.asarray(cycles))


def run(seeds=DEFAULT_SEEDS, device: str | None = None,
        batch_size: int = 512) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct every occlusion window with the frozen formal checkpoints.

    Cycles are processed in batches (the formal evaluation batch size), so the
    whole grid is a few thousand forward passes rather than one per cycle.
    """
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    C.log("=" * 78); C.log("Sliding-window occlusion + support-vulnerability"); C.log("=" * 78)

    certified = C.module()
    train = certified.load_clean(DATASET, "train")
    vali = certified.load_clean(DATASET, "vali")
    test = certified.load_clean(DATASET, "test")
    spline, model, s_ind, _, _ = certified.select_manifold(train, vali)
    state = certified.prepare_test_decomposition(test, spline, model)
    eligible = state["eligible"]
    identity = list(zip(np.asarray(test["cells"]).astype(str),
                        np.asarray(test["cycles"], dtype=int)))
    position = {key: i for i, key in enumerate(identity)}

    dataset = BatteryDataset(DATASET, split="test", condition="clean")
    voltage, current, cells, cycles = _load_cycles(dataset, 1000)
    length = voltage.shape[1]
    definitions = window_definitions(length, WIDTHS)

    # keep only certified-eligible cycles, in the certified identity order
    order = np.array([position[(c, int(y))] for c, y in zip(cells, cycles)])
    keep = eligible[order]
    voltage, current = voltage[keep], current[keep]
    order = order[keep]
    eligible_position = {(str(cell), int(cycle)): index
                         for index, (cell, cycle) in enumerate(zip(cells[keep], cycles[keep]))}
    C.log(f"    {len(voltage)} eligible cycles x {len(definitions)} windows x {len(seeds)} seeds")

    clean_metrics, _ = certified.evaluate_reconstruction(
        test["voltage"], test, state, spline, model, s_ind, train["soh"])
    clean_lookup = {(str(row.cell), int(row.cycle)): row
                    for _, row in clean_metrics.iterrows()}

    rows, metric_rows = [], []
    for seed in seeds:
        args = _args(seed)
        runner = Exp_Reconstruction_Stress_Test(args)
        runner._load_best_checkpoint(
            REPOSITORY_ROOT / "source_code" / "checkpoints" / "Reconstruction" /
            DATASET / f"Reconstruction_seed{seed}_batterydit_{DATASET}_0")
        runner.model.eval()

        for definition in definitions:
            mask = np.ones(length, dtype=np.float32)
            mask[definition["start_index"]:definition["end_index_exclusive"]] = 0.0
            predictions = np.zeros_like(voltage)
            with torch.no_grad():
                for start in range(0, len(voltage), batch_size):
                    stop = min(start + batch_size, len(voltage))
                    v = torch.tensor(voltage[start:stop] * mask, device=device,
                                     dtype=torch.float32).unsqueeze(-1)
                    i = torch.tensor(current[start:stop] * mask, device=device,
                                     dtype=torch.float32).unsqueeze(-1)
                    predictions[start:stop] = runner._predict_physical(v, i).cpu().numpy().reshape(
                        stop - start, -1)
            rmse = np.sqrt(((predictions - voltage) ** 2).mean(axis=1))
            waveform_damage = np.sqrt(
                ((voltage * (1.0 - mask)) ** 2).mean(axis=1))
            for j in range(len(voltage)):
                cell, cycle = identity[order[j]]
                rows.append({"cell": cell, "cycle": int(cycle), "model_seed": seed,
                             "window_id": definition["window_id"],
                             "width_fraction": definition["width_fraction"],
                             "window_start": definition["window_start"],
                             "waveform_RMSE": float(rmse[j])})

            # certified metrics for this window.  `evaluate_reconstruction`
            # indexes the FULL test block with `state["eligible"]`, so the
            # predictions are scattered back into a full-length array first.
            shape = np.asarray(test["voltage"]).shape   # (N, 1000, 1)
            full = np.zeros(shape, dtype=np.float32)
            full[eligible] = predictions.astype(np.float32).reshape(len(predictions), -1,
                                                                    shape[-1])
            metrics, _ = certified.evaluate_reconstruction(
                full, test, state, spline, model, s_ind, train["soh"])
            for cell, cycle, e_pop, e_ind, q_ratio, c_ind, waveform in zip(
                    metrics.cell, metrics.cycle, metrics.E_pop, metrics.E_ind,
                    metrics.Q, metrics.C_ind, metrics.waveform_rmse):
                key = (str(cell), int(cycle))
                clean = clean_lookup[key]
                row_index = eligible_position[key]
                metric_rows.append({
                    "window_id": definition["window_id"], "model_seed": seed,
                    "cell": cell, "cycle": int(cycle),
                    "width_fraction": float(definition["width_fraction"]),
                    "window_start": float(definition["window_start"]),
                    "waveform_damage": float(waveform_damage[row_index]),
                    "waveform_RMSE": float(waveform),
                    "E_pop_clean": float(clean.E_pop), "E_pop_occluded": float(e_pop),
                    "E_ind_clean": float(clean.E_ind), "E_ind_occluded": float(e_ind),
                    "Q_clean": float(clean.Q), "Q_occluded": float(q_ratio),
                    "C_ind_clean": float(clean.C_ind), "C_ind_occluded": float(c_ind),
                    "Delta_E_pop": float(e_pop - clean.E_pop),
                    "Delta_E_ind": float(e_ind - clean.E_ind),
                    "C_ind_loss": float(clean.C_ind - c_ind),
                })
        del runner

    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def _args(seed: int):
    """Formal argument namespace for one frozen BatteryDiT checkpoint.

    Uses the repository's own entry-point parser so every architecture and
    protocol default is exactly the formal one; only the identity fields are
    overridden.
    """
    from main import parser as formal_parser  # type: ignore

    args, _ = formal_parser().parse_known_args([
        "--task_name", "Reconstruction",
        "--is_training", "0",
        "--recon_model", "batterydit",
        "--data", DATASET,
        "--seed", str(seed),
        "--use_gpu", "1",
        "--gpu", "0",
        "--model_id", f"seed{seed}",
    ])
    args.recn_chkpt = str(REPOSITORY_ROOT / "source_code" / "checkpoints" /
                          "Reconstruction" / DATASET /
                          f"Reconstruction_seed{seed}_batterydit_{DATASET}_0" /
                          "checkpoint.pth")
    return args
