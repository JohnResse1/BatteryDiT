"""Frozen reconstruction followed by the formal TRB SOH/RUL predictor."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider, paired_data_provider
from data_provider.data_loader import BatteryDataset
from models.checkpoint_loading import load_reconstruction_checkpoint, state_dict_from_checkpoint
from models.predictors.trb.model import Model as TRB
from models.predictors.tcn_reconstruction import Model as ReconstructionTCN
from utils.benchmark_corruptions import corrupt_training_inputs
from utils.metrics import metric

from .exp_basic import Exp_Basic
from .protocol import FORMAL_RECONSTRUCTION_CONDITIONS, PROTOCOL_NAME
from .reconstruction import SOURCE_CODE_ROOT


class Exp_Predict_Downstream(Exp_Basic):
    """Train the formal predictor on frozen reconstructions and report SOH metrics."""

    def _build_model(self) -> nn.Module:
        predictor = str(getattr(self.args, "predictor", "trb")).lower()
        if predictor == "trb":
            return TRB(self.args, dropout=float(getattr(self.args, "dropout", 0.1)))
        if predictor == "tcn":
            return ReconstructionTCN(self.args, dropout=float(getattr(self.args, "dropout", 0.1)))
        raise ValueError(f"unknown reconstruction-assisted predictor {predictor!r}; choose trb or tcn")

    def __init__(self, args: Any):
        super().__init__(args)
        recon_name = str(args.recon_model).lower()
        if recon_name not in Exp_Basic.model_dict:
            raise ValueError(f"unknown formal reconstruction model {recon_name!r}")
        self.recon_model = Exp_Basic.model_dict[recon_name](args).to(self.device)
        self._load_reconstructor()
        self.recon_model.eval()
        for parameter in self.recon_model.parameters():
            parameter.requires_grad = False
        self._csdi_evaluation_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}

    def _run_paths(self, setting: str) -> tuple[Path, Path]:
        checkpoints = (
            SOURCE_CODE_ROOT / Path(getattr(self.args, "checkpoints", "checkpoints"))
            / str(self.args.task_name) / str(self.args.data) / setting
        )
        result_group = str(getattr(self.args, "_result_group", "")).strip() or str(self.args.task_name)
        results = (
            SOURCE_CODE_ROOT / Path(getattr(self.args, "results_dir", "../results"))
            / result_group / str(self.args.data) / setting
        )
        checkpoints.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        return checkpoints, results

    def _load_reconstructor(self) -> None:
        path = str(getattr(self.args, "recn_chkpt", "")).strip()
        if not path and str(self.args.recon_model).lower() != "kalman":
            raise ValueError("--recn_chkpt is required for downstream prediction")
        if path:
            load_reconstruction_checkpoint(self.recon_model, path, map_location=self.device, strict=True)

    @staticmethod
    def _target(soh: torch.Tensor, rul: torch.Tensor) -> torch.Tensor:
        return torch.cat([soh.reshape(soh.size(0), 1), rul.reshape(rul.size(0), 1)], dim=1)

    @torch.no_grad()
    def _reconstruct(self, voltage: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        normalized = self.recon_model(V=voltage, I=current, mode="recon_only_sampling", steps=int(getattr(self.args, "sampling_steps", 1)))
        return self.recon_model._denormalize_phy(normalized)

    def _corrupt_training_inputs_for_trb(
        self, voltage: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the benchmark corruption mixture before frozen recon → TRB.

        This is only the downstream predictor's augmentation.  Reconstruction
        models are trained separately on the clean training view.
        """
        return corrupt_training_inputs(voltage, current, self.args)

    def _evaluate(self, split: str, condition: str) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        cached = self._cached_csdi_evaluation(split, condition)
        if cached is not None:
            reconstruction, target, keys = cached
            self.model.eval()
            # The cached CSDI path holds the whole split in one tensor, so it
            # must be pushed through the predictor in evaluation-sized chunks.
            # A single forward pass over e.g. 54,177 cycles makes the global
            # attention allocate ~13 GiB and OOMs a 24 GiB card; chunking is
            # numerically identical (all predictor ops are per-sample).
            batch_size = int(getattr(self.args, "eval_batch_size", None) or
                             max(1, int(self.args.batch_size) * 8))
            chunks = []
            with torch.no_grad():
                for start in range(0, reconstruction.shape[0], batch_size):
                    chunk = reconstruction[start:start + batch_size].to(self.device)
                    chunks.append(self.model(chunk).cpu())
            prediction = torch.cat(chunks).numpy()
            target_array = target.numpy()
            mae, mse, rmse = metric(prediction[:, 0], target_array[:, 0])
            return {"sample_count": int(target_array.shape[0]), "soh_mae": float(mae), "soh_mse": float(mse), "soh_rmse": float(rmse)}, {"prediction": prediction, "target": target_array, "keys": keys}
        loader = paired_data_provider(self.args, split, condition)[1]
        predictions, targets, sample_keys = [], [], []
        self.model.eval()
        with torch.no_grad():
            for current, voltage, _, soh, rul, batch_keys in loader:
                current, voltage = current.to(self.device), voltage.to(self.device)
                reconstruction = self._reconstruct(voltage, current)
                predictions.append(self.model(reconstruction).cpu().numpy())
                targets.append(self._target(soh, rul).numpy())
                sample_keys.extend(str(key) for key in batch_keys)
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        mae, mse, rmse = metric(prediction[:, 0], target[:, 0])
        return {
            "sample_count": int(target.shape[0]),
            "soh_mae": float(mae),
            "soh_mse": float(mse),
            "soh_rmse": float(rmse),
        }, {"prediction": prediction, "target": target, "keys": np.asarray(sample_keys, dtype=str)}

    def _cached_csdi_evaluation(self, split: str, condition: str):
        """Freeze stochastic CSDI reconstructions for fair model selection."""
        if str(self.args.recon_model).lower() != "csdi":
            return None
        cache_key = (split, condition)
        if cache_key in self._csdi_evaluation_cache:
            return self._csdi_evaluation_cache[cache_key]
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        seed = int(getattr(self.args, "eval_reconstruction_seed", 3407))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            loader = paired_data_provider(self.args, split, condition)[1]
            reconstructions, targets, sample_keys = [], [], []
            self.recon_model.eval()
            with torch.no_grad():
                for current, voltage, _, soh, rul, batch_keys in loader:
                    reconstructions.append(self._reconstruct(voltage.to(self.device), current.to(self.device)).cpu())
                    targets.append(self._target(soh, rul))
                    sample_keys.extend(str(sample_key) for sample_key in batch_keys)
            self._csdi_evaluation_cache[cache_key] = (
                torch.cat(reconstructions), torch.cat(targets), np.asarray(sample_keys, dtype=str)
            )
        finally:
            random.setstate(python_state); np.random.set_state(numpy_state); torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return self._csdi_evaluation_cache[cache_key]

    def _kalman_training_loader(self) -> DataLoader:
        """Build the one fixed Kalman training feature set used by all seeds."""
        root = SOURCE_CODE_ROOT / Path(self.args.checkpoints) / "Reconstruction" / "kalman" / str(self.args.data)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "fixed_training_features.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                reconstruction = torch.from_numpy(np.asarray(archive["reconstruction"], dtype=np.float32))
                target = torch.from_numpy(np.asarray(archive["target"], dtype=np.float32))
        else:
            dataset = BatteryDataset(self.args.data, split="train", condition="clean")
            source = DataLoader(dataset, batch_size=int(self.args.batch_size), shuffle=False, num_workers=int(self.args.num_workers))
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            random.seed(3407); np.random.seed(3407); torch.manual_seed(3407)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(3407)
            try:
                reconstructions, targets = [], []
                with torch.no_grad():
                    for current, voltage, soh, rul in source:
                        current, voltage = current.to(self.device), voltage.to(self.device)
                        corrupted_voltage, corrupted_current = corrupt_training_inputs(voltage, current, self.args)
                        reconstructions.append(self._reconstruct(corrupted_voltage, corrupted_current).cpu())
                        targets.append(self._target(soh, rul))
                reconstruction, target = torch.cat(reconstructions), torch.cat(targets)
                np.savez_compressed(path, reconstruction=reconstruction.numpy(), target=target.numpy(), augmentation_seed=np.asarray([3407], dtype=np.int32))
            finally:
                random.setstate(python_state); np.random.set_state(numpy_state); torch.random.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
        generator = torch.Generator().manual_seed(int(self.args.seed))
        return DataLoader(TensorDataset(reconstruction, target), batch_size=int(self.args.batch_size), shuffle=True, num_workers=int(self.args.num_workers), generator=generator)

    def train(self, setting: str):
        checkpoints, results = self._run_paths(setting)
        train_loader = self._kalman_training_loader() if str(self.args.recon_model).lower() == "kalman" else data_provider(self.args, "train", error_type="clean")[1]
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.args.learning_rate), weight_decay=float(getattr(self.args, "weight_decay", 1e-4)))
        criterion = nn.SmoothL1Loss()
        best_rmse = float("inf")
        history = []
        for epoch in range(1, int(self.args.epochs) + 1):
            self.model.train()
            losses = []
            for batch in train_loader:
                if str(self.args.recon_model).lower() == "kalman":
                    reconstruction, target = (value.to(self.device) for value in batch)
                else:
                    current, voltage, soh, rul = batch
                    current, voltage = current.to(self.device), voltage.to(self.device)
                    augmented_voltage, augmented_current = self._corrupt_training_inputs_for_trb(voltage, current)
                    reconstruction = self._reconstruct(augmented_voltage, augmented_current)
                    target = self._target(soh, rul).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(reconstruction), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            is_validation_epoch = epoch % int(getattr(self.args, "validation_interval", 10)) == 0 or epoch == int(self.args.epochs)
            if is_validation_epoch:
                validation = {condition: self._evaluate("vali", condition)[0] for condition in FORMAL_RECONSTRUCTION_CONDITIONS}
                selection_rmse = float(np.mean([item["soh_rmse"] for item in validation.values()]))
                record["validation"] = validation
                record["validation_mean_soh_rmse"] = selection_rmse
            history.append(record)
            payload = {"model": self._unwrap(self.model).state_dict(), "epoch": epoch, "record": record}
            torch.save(payload, checkpoints / "last.pth")
            if is_validation_epoch and selection_rmse < best_rmse:
                best_rmse = selection_rmse
                torch.save(payload, checkpoints / "best.pth")
            (results / "history.json").write_text(json.dumps(history, indent=2))
        (results / "config.json").write_text(json.dumps({"protocol": PROTOCOL_NAME, "reconstruction_checkpoint": str(self.args.recn_chkpt), "arguments": {key: value for key, value in vars(self.args).items() if isinstance(value, (str, int, float, bool))}}, indent=2))
        return self.model

    def vali(self, setting: str):
        _, results = self._run_paths(setting)
        reports = {condition: self._evaluate("vali", condition)[0] for condition in FORMAL_RECONSTRUCTION_CONDITIONS}
        reports["mean_soh_rmse"] = float(np.mean([item["soh_rmse"] for item in reports.values()]))
        (results / "validation_metrics.json").write_text(json.dumps(reports, indent=2))
        return reports

    def _load_predictor(self, path: str | Path) -> None:
        """Load one frozen downstream predictor checkpoint strictly."""
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"predictor checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location=self.device)
        state = payload.get("model", state_dict_from_checkpoint(payload)) if isinstance(payload, dict) else payload
        self._unwrap(self.model).load_state_dict(state, strict=True)

    def test(self, setting: str):
        checkpoints, results = self._run_paths(setting)
        path = checkpoints / "best.pth"
        self._load_predictor(path)
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
