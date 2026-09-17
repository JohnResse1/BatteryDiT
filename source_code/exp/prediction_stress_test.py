"""Evaluation-only SOH stress test for a frozen reconstructor and TRB."""
from __future__ import annotations

import json

import numpy as np
import torch

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.stress_corruptions import (
    FORMAL_STRESS_PROTOCOL, STRESS_CONDITIONS, STRESS_SEED_SCHEME,
    apply_stress_corruption, reset_stress_rng,
)

from .downstream_prediction import Exp_Predict_Downstream
from models.checkpoint_loading import state_dict_from_checkpoint


class Exp_Predict_Stress_Test(Exp_Predict_Downstream):
    """Run the frozen stress protocol and report SOH-only metrics."""

    def train(self, setting: str):
        raise ValueError("PredictStressTest is evaluation-only; use --is_training 0")

    def vali(self, setting: str):
        raise ValueError("PredictStressTest has no validation phase")

    def _load_predictor(self) -> None:
        path = str(getattr(self.args, "pred_chkpt", "")).strip()
        if not path:
            raise ValueError("--pred_chkpt is required for prediction stress testing")
        payload = torch.load(path, map_location=self.device)
        state = payload.get("model", state_dict_from_checkpoint(payload)) if isinstance(payload, dict) else payload
        self._unwrap(self.model).load_state_dict(state, strict=True)

    def _evaluate_level(self, loader, corrupt):
        predictions, targets = [], []
        self.model.eval()
        with torch.no_grad():
            for current, voltage, soh, _ in loader:
                current, voltage = current.to(self.device), voltage.to(self.device)
                corrupted_voltage, corrupted_current = corrupt(voltage, current)
                predictions.append(self.model(self._reconstruct(corrupted_voltage, corrupted_current))[:, 0].cpu().numpy())
                targets.append(soh.reshape(-1).numpy())
        prediction, target = np.concatenate(predictions), np.concatenate(targets)
        mae, mse, rmse = metric(prediction, target)
        return {"sample_count": int(target.size), "soh_mae": float(mae), "soh_mse": float(mse), "soh_rmse": float(rmse)}

    def test(self, setting: str):
        _, results_dir = self._run_paths(setting)
        self._load_predictor()
        loader = data_provider(self.args, "test", error_type="clean")[1]
        model_seed = int(getattr(self.args, "seed", 2021))
        evaluation_seed = int(getattr(self.args, "stress_seed", 3407))
        results = {
            "protocol": FORMAL_STRESS_PROTOCOL,
            "seed_scheme": STRESS_SEED_SCHEME,
            "model_seed": model_seed,
            "evaluation_seed": evaluation_seed,
            "reconstruction_model": str(self.args.recon_model).lower(),
            "predictor": str(getattr(self.args, "predictor", "trb")).lower(),
            "metric_scope": "soh_only",
        }
        for name in STRESS_CONDITIONS:
            results[name] = []
            for index in range(5):
                reset_stress_rng(evaluation_seed, name, index)
                results[name].append(self._evaluate_level(
                    loader,
                    lambda voltage, current, condition=name, level=index:
                        apply_stress_corruption(condition, level, voltage, current),
                ))
        (results_dir / "prediction_stress_metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
        return results
