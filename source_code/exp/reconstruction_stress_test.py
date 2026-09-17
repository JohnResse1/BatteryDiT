"""Evaluation-only reconstruction stress test for a formal checkpoint."""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
import torch

from data_provider.data_factory import data_provider
from utils.stress_corruptions import (
    FORMAL_STRESS_PROTOCOL, STRESS_CONDITIONS, STRESS_SEED_SCHEME,
    apply_stress_corruption, reset_stress_rng,
)

from .reconstruction import Exp_Reconstruction


class Exp_Reconstruction_Stress_Test(Exp_Reconstruction):
    """Measure a frozen reconstructor on five fixed corruption severities."""

    def train(self, setting: str):
        raise ValueError("ReconstructionStressTest is evaluation-only; use --is_training 0")

    def vali(self, setting: str):
        raise ValueError("ReconstructionStressTest has no validation phase")

    def _evaluate_level(self, loader, corrupt: Callable):
        values = []
        self.model.eval()
        with torch.no_grad():
            for current, voltage, _, _ in loader:
                current, voltage = current.to(self.device), voltage.to(self.device)
                corrupted_voltage, corrupted_current = corrupt(voltage, current)
                reconstruction = self._predict_physical(corrupted_voltage, corrupted_current)
                values.append((reconstruction - voltage).square().mean(dim=(1, 2)).sqrt().cpu().numpy())
        per_sample = np.concatenate(values)
        return {
            "sample_count": int(per_sample.size),
            "sample_rmse_mean": float(per_sample.mean()),
            "sample_rmse_std": float(per_sample.std(ddof=1)) if per_sample.size > 1 else 0.0,
        }

    def test(self, setting: str):
        checkpoints, results_dir = self._run_paths(setting)
        self._load_best_checkpoint(checkpoints)
        loader = data_provider(self.args, "test", error_type="clean")[1]
        model_seed = int(getattr(self.args, "seed", 2021))
        evaluation_seed = int(getattr(self.args, "stress_seed", 3407))
        results = {
            "protocol": FORMAL_STRESS_PROTOCOL,
            "seed_scheme": STRESS_SEED_SCHEME,
            "model_seed": model_seed,
            "evaluation_seed": evaluation_seed,
            "reconstruction_model": str(self.args.recon_model).lower(),
            **{condition: [] for condition in STRESS_CONDITIONS},
        }
        for condition in STRESS_CONDITIONS:
            for level_index in range(5):
                reset_stress_rng(evaluation_seed, condition, level_index)
                results[condition].append(self._evaluate_level(
                    loader,
                    lambda voltage, current, name=condition, level=level_index:
                        apply_stress_corruption(name, level, voltage, current),
                ))
        (results_dir / "stress_metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
        return results
