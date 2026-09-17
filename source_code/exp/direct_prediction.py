"""Formal direct SOH prediction benchmark without trajectory reconstruction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from data_provider.data_factory import data_provider, paired_data_provider
from models.predictors.direct import GRU, LSTM, TCN, Transformer
from models.predictors.trb.model import Model as TRB
from utils.benchmark_corruptions import corrupt_training_inputs
from utils.metrics import metric

from .protocol import FORMAL_RECONSTRUCTION_CONDITIONS, PROTOCOL_NAME
from .reconstruction import SOURCE_CODE_ROOT


DIRECT_MODELS: dict[str, type[nn.Module]] = {
    "lstm": LSTM,
    "gru": GRU,
    "transformer": Transformer,
    "tcn": TCN,
    "trb": TRB,
}


class Exp_Direct_Prediction:
    """Train a raw-trajectory baseline under the formal five-view protocol."""

    def __init__(self, args: Any):
        self.args = args
        self.device = self._acquire_device()
        self.name = str(args.direct_model).lower()
        try:
            self.model = DIRECT_MODELS[self.name](args).to(self.device)
        except KeyError as error:
            raise ValueError(f"unknown direct predictor {self.name!r}; choose {', '.join(DIRECT_MODELS)}") from error

    def _acquire_device(self) -> torch.device:
        if bool(getattr(self.args, "use_gpu", False)) and torch.cuda.is_available():
            return torch.device(f"cuda:{int(getattr(self.args, 'gpu', 0))}")
        return torch.device("cpu")

    def _run_paths(self, setting: str) -> tuple[Path, Path]:
        checkpoints = SOURCE_CODE_ROOT / Path(self.args.checkpoints) / "DirectPrediction" / str(self.args.data) / setting
        results = SOURCE_CODE_ROOT / Path(self.args.results_dir) / "DirectPrediction" / str(self.args.data) / setting
        checkpoints.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        return checkpoints, results

    @staticmethod
    def _target(soh: torch.Tensor, rul: torch.Tensor) -> torch.Tensor:
        return torch.cat([soh.reshape(-1, 1), rul.reshape(-1, 1)], dim=1)

    def _input(self, voltage: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return voltage.squeeze(-1) if self.name == "trb" else torch.cat([voltage, current], dim=-1)

    def _evaluate(self, split: str, condition: str) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        loader = paired_data_provider(self.args, split, condition)[1]
        predictions, targets, sample_keys = [], [], []
        self.model.eval()
        with torch.no_grad():
            for current, voltage, _, soh, rul, batch_keys in loader:
                current, voltage = current.to(self.device), voltage.to(self.device)
                predictions.append(self.model(self._input(voltage, current)).cpu().numpy())
                targets.append(self._target(soh, rul).numpy())
                sample_keys.extend(str(key) for key in batch_keys)
        prediction, target = np.concatenate(predictions), np.concatenate(targets)
        mae, mse, rmse = metric(prediction[:, 0], target[:, 0])
        return {"sample_count": int(target.shape[0]), "soh_mae": float(mae), "soh_mse": float(mse), "soh_rmse": float(rmse)}, {"prediction": prediction, "target": target, "keys": np.asarray(sample_keys, dtype=str)}

    def train(self, setting: str):
        checkpoints, results = self._run_paths(setting)
        loader = data_provider(self.args, "train", error_type="clean")[1]
        optimizer = optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate), weight_decay=float(self.args.weight_decay))
        criterion, best_rmse, history = nn.SmoothL1Loss(), float("inf"), []
        for epoch in range(1, int(self.args.epochs) + 1):
            self.model.train()
            losses = []
            for current, voltage, soh, rul in loader:
                current, voltage = current.to(self.device), voltage.to(self.device)
                corrupted_voltage, corrupted_current = corrupt_training_inputs(voltage, current, self.args)
                target = self._target(soh, rul).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(self._input(corrupted_voltage, corrupted_current)), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            record: dict[str, Any] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if epoch % int(self.args.validation_interval) == 0 or epoch == int(self.args.epochs):
                validation = {condition: self._evaluate("vali", condition)[0] for condition in FORMAL_RECONSTRUCTION_CONDITIONS}
                selection = float(np.mean([item["soh_rmse"] for item in validation.values()]))
                record.update(validation=validation, validation_mean_soh_rmse=selection)
                if selection < best_rmse:
                    best_rmse = selection
                    torch.save({"model": self.model.state_dict(), "epoch": epoch, "record": record}, checkpoints / "best.pth")
            torch.save({"model": self.model.state_dict(), "epoch": epoch, "record": record}, checkpoints / "last.pth")
            history.append(record)
            (results / "history.json").write_text(json.dumps(history, indent=2))
        (results / "config.json").write_text(json.dumps({"protocol": PROTOCOL_NAME, "model": self.name, "arguments": {key: value for key, value in vars(self.args).items() if isinstance(value, (str, int, float, bool))}}, indent=2))

    def vali(self, setting: str):
        _, results = self._run_paths(setting)
        reports = {condition: self._evaluate("vali", condition)[0] for condition in FORMAL_RECONSTRUCTION_CONDITIONS}
        reports["mean_soh_rmse"] = float(np.mean([item["soh_rmse"] for item in reports.values()]))
        (results / "validation_metrics.json").write_text(json.dumps(reports, indent=2))
        return reports

    def test(self, setting: str):
        checkpoints, results = self._run_paths(setting)
        payload = torch.load(checkpoints / "best.pth", map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        reports, arrays = {}, {}
        for condition in FORMAL_RECONSTRUCTION_CONDITIONS:
            report, values = self._evaluate("test", condition)
            reports[condition] = report
            arrays[f"{condition}_prediction"] = values["prediction"]
            arrays[f"{condition}_target"] = values["target"]
            arrays[f"{condition}_keys"] = values["keys"]
        (results / "test_metrics.json").write_text(json.dumps({"protocol": PROTOCOL_NAME, "conditions": reports}, indent=2))
        np.savez_compressed(results / "predictions.npz", **arrays)
        return reports
