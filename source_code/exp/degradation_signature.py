"""Read out SOH-related ordering from frozen TRB degradation representations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from data_provider.data_loader import BatteryDataset
from utils.stress_corruptions import (
    FORMAL_STRESS_PROTOCOL, STRESS_CONDITIONS, STRESS_SEED_SCHEME,
    apply_stress_corruption, reset_stress_rng,
)

from .downstream_prediction import Exp_Predict_Downstream
from .reconstruction import SOURCE_CODE_ROOT


TOPOLOGIES = STRESS_CONDITIONS


def _balanced_indices(dataset: BatteryDataset, cycles_per_cell: int) -> np.ndarray:
    if cycles_per_cell < 1:
        raise ValueError("cycles_per_cell must be positive")
    cells = np.asarray([path.relative_to(dataset.root).parts[0] for path in dataset.paths])
    selected: list[int] = []
    for cell in dict.fromkeys(cells.astype(str)):
        members = np.flatnonzero(cells == cell)
        count = min(len(members), cycles_per_cell)
        positions = np.unique(np.rint(np.linspace(0, len(members) - 1, count)).astype(int))
        selected.extend(members[positions].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _balanced_loader(args, split: str) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    dataset = BatteryDataset(args.data, split=split, condition="clean")
    indices = _balanced_indices(dataset, int(args.cycles_per_cell))
    paths = [dataset.paths[index] for index in indices]
    keys = np.asarray([str(path.relative_to(dataset.root)) for path in paths])
    cells = np.asarray([key.split("/", 1)[0] for key in keys])
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=int(getattr(args, "eval_batch_size", None) or args.batch_size * 8),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, keys, cells


def _tau(target: np.ndarray, score: np.ndarray) -> float:
    value = float(kendalltau(target, score, variant="b").statistic)
    if not np.isfinite(value):
        raise ValueError("Kendall tau-b is undefined for the selected samples")
    return value


class Exp_Degradation_Signature(Exp_Predict_Downstream):
    """Apply a clean-training Ridge probe to clean, corrupted, and recovered latents."""

    def __init__(self, args):
        if str(args.recon_model).lower() != "batterydit" or str(args.predictor).lower() != "trb":
            raise ValueError("DegradationSignature is defined for frozen BatteryDiT and TRB checkpoints")
        super().__init__(args)

    def train(self, setting: str):
        raise ValueError("DegradationSignature is evaluation-only; use --is_training 0")

    def vali(self, setting: str):
        raise ValueError("DegradationSignature has no validation phase")

    def _results_path(self, setting: str) -> Path:
        root = Path(getattr(self.args, "results_dir", "../results"))
        root = root if root.is_absolute() else SOURCE_CODE_ROOT / root
        output = root / str(self.args.task_name) / str(self.args.data) / setting
        output.mkdir(parents=True, exist_ok=True)
        return output

    @torch.inference_mode()
    def _clean_latents(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        latent, soh = [], []
        self.model.eval()
        for _, voltage, batch_soh, _ in loader:
            latent.append(self.model.encode(voltage.to(self.device)).cpu().numpy())
            soh.append(batch_soh.reshape(-1).numpy())
        return np.concatenate(latent), np.concatenate(soh)

    @torch.inference_mode()
    def _condition_latents(self, loader: DataLoader, topology: str, level: int, evaluation_seed: int):
        reset_stress_rng(evaluation_seed, topology, level)
        corrupted_latent, reconstructed_latent, soh = [], [], []
        self.model.eval()
        self.recon_model.eval()
        for current, voltage, batch_soh, _ in loader:
            current, voltage = current.to(self.device), voltage.to(self.device)
            corrupted_voltage, corrupted_current = apply_stress_corruption(topology, level, voltage, current)
            reconstruction = self._reconstruct(corrupted_voltage, corrupted_current)
            corrupted_latent.append(self.model.encode(corrupted_voltage).cpu().numpy())
            reconstructed_latent.append(self.model.encode(reconstruction).cpu().numpy())
            soh.append(batch_soh.reshape(-1).numpy())
        return np.concatenate(corrupted_latent), np.concatenate(reconstructed_latent), np.concatenate(soh)

    def test(self, setting: str):
        self._load_predictor()
        train_loader, train_keys, train_cells = _balanced_loader(self.args, "train")
        test_loader, test_keys, test_cells = _balanced_loader(self.args, "test")
        train_latent, train_soh = self._clean_latents(train_loader)
        clean_latent, test_soh = self._clean_latents(test_loader)

        scaler = StandardScaler().fit(train_latent)
        train_scaled = scaler.transform(train_latent)
        clean_scaled = scaler.transform(clean_latent)
        probe = Ridge(alpha=1.0).fit(train_scaled, train_soh)
        pca = PCA(n_components=2, random_state=int(self.args.seed)).fit(train_scaled)
        clean_tau = _tau(test_soh, probe.predict(clean_scaled))

        corrupted_tau = np.empty((len(TOPOLOGIES), 5), dtype=np.float64)
        reconstructed_tau = np.empty_like(corrupted_tau)
        representative_corrupted = representative_reconstructed = None
        evaluation_seed = int(getattr(self.args, "stress_seed", 3407))
        for topology_index, topology in enumerate(TOPOLOGIES):
            for level in range(5):
                corrupted_latent, reconstructed_latent, condition_soh = self._condition_latents(
                    test_loader, topology, level, evaluation_seed,
                )
                corrupted_scaled = scaler.transform(corrupted_latent)
                reconstructed_scaled = scaler.transform(reconstructed_latent)
                corrupted_tau[topology_index, level] = _tau(condition_soh, probe.predict(corrupted_scaled))
                reconstructed_tau[topology_index, level] = _tau(condition_soh, probe.predict(reconstructed_scaled))
                if topology == "hybrid" and level == 2:
                    representative_corrupted = pca.transform(corrupted_scaled)
                    representative_reconstructed = pca.transform(reconstructed_scaled)
        if representative_corrupted is None or representative_reconstructed is None:
            raise RuntimeError("Hybrid L3 representation was not collected")

        output = self._results_path(setting)
        np.savez_compressed(
            output / "degradation_signature.npz",
            protocol=FORMAL_STRESS_PROTOCOL,
            seed=np.int32(self.args.seed),
            evaluation_seed=np.int32(evaluation_seed),
            topologies=np.asarray(TOPOLOGIES),
            levels=np.arange(1, 6, dtype=np.int32),
            cycles_per_cell=np.int32(self.args.cycles_per_cell),
            train_keys=train_keys,
            test_keys=test_keys,
            train_cell_count=np.int32(len(set(train_cells))),
            test_cell_count=np.int32(len(set(test_cells))),
            clean_tau=np.float64(clean_tau),
            corrupted_tau=corrupted_tau,
            reconstructed_tau=reconstructed_tau,
            representative_condition=np.asarray("hybrid_L3"),
            representative_pca_clean=pca.transform(clean_scaled).astype(np.float32),
            representative_pca_corrupted=representative_corrupted.astype(np.float32),
            representative_pca_reconstructed=representative_reconstructed.astype(np.float32),
            representative_soh=test_soh.astype(np.float32),
            representative_keys=test_keys,
            pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        )
        summary = {
            "protocol": FORMAL_STRESS_PROTOCOL,
            "seed_scheme": STRESS_SEED_SCHEME,
            "model_seed": int(self.args.seed),
            "evaluation_seed": evaluation_seed,
            "reconstruction_model": "batterydit",
            "predictor": "trb",
            "cycles_per_cell": int(self.args.cycles_per_cell),
            "train_samples": int(len(train_keys)),
            "test_samples": int(len(test_keys)),
            "clean_tau": clean_tau,
            "mean_corrupted_tau": float(corrupted_tau.mean()),
            "mean_reconstructed_tau": float(reconstructed_tau.mean()),
        }
        (output / "config.json").write_text(json.dumps({"arguments": {"task_name": self.args.task_name, "data": self.args.data, "seed": int(self.args.seed), "recon_model": "batterydit", "predictor": "trb", "stress_seed": evaluation_seed, "cycles_per_cell": int(self.args.cycles_per_cell)}}, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
