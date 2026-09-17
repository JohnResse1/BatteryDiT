"""Formal reconstruction experiment."""
from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from data_provider.data_factory import data_provider, paired_data_provider
from data_provider.data_loader import DATA_ROOT, PairedBatteryDataset
from models.checkpoint_loading import load_reconstruction_state_dict

from .exp_basic import Exp_Basic
from .protocol import (
    BATTERYDIT_GRADIENT_LOSS_WEIGHT,
    FORMAL_RECONSTRUCTION_CONDITIONS,
    PROTOCOL_NAME,
    RECONSTRUCTION_TRAINING,
)

SOURCE_CODE_ROOT = Path(__file__).resolve().parents[1]


def voltage_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    voltage_range: float,
) -> torch.Tensor:
    '''Compute the voltage gradient loss between prediction and target.'''
    return F.mse_loss(
        (prediction[:, 1:] - prediction[:, :-1]) * voltage_range,
        (target[:, 1:] - target[:, :-1]) * voltage_range,
    )


def _root(args: Any, name: str, default: str) -> Path:
    value = Path(getattr(args, name, default))
    return value if value.is_absolute() else SOURCE_CODE_ROOT / value


def _json_args(args: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


class Exp_Reconstruction(Exp_Basic):
    """Train and evaluate one formal reconstruction model.

    BatteryDiT always uses the fixed objective: pointwise velocity MSE plus physical-voltage gradient MSE with weight 1.0.
    CSDI and Transformer use their own formal training objectives;
    Kalman is eval-only.  This runner trains reconstruction only: it never
    instantiates TRB or applies the downstream corruption mixture.
    """

    def _run_paths(self, setting: str) -> tuple[Path, Path]:
        dataset = str(self.args.data)
        task = str(getattr(self.args, "task_name", "Reconstruction"))
        checkpoints = _root(self.args, "checkpoints", "checkpoints") / task / dataset / setting
        result_group = str(getattr(self.args, "_result_group", "")).strip() or task
        results = _root(self.args, "results_dir", "../results") / result_group / dataset / setting
        checkpoints.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        return checkpoints, results

    def _write_config(self, results: Path) -> None:
        model_name = str(self.args.recon_model).lower()
        metadata = {
            "protocol": PROTOCOL_NAME,
            "model": model_name,
            "batterydit_loss": (
                "pointwise velocity MSE + 1.0 * physical-voltage gradient MSE"
                if model_name == "batterydit" else None
            ),
            "training": RECONSTRUCTION_TRAINING.get(model_name, {"trainable": False}),
            "arguments": _json_args(self.args),
        }
        (results / "config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _loader(self, split: str):
        return data_provider(self.args, split, error_type="clean")[1]

    def _paired_loader(self, split: str, condition: str):
        return paired_data_provider(self.args, split, condition)[1]

    def _selection_loader(self, condition: str) -> DataLoader:
        """Return the fixed battery-balanced validation subset.

        Every validation battery contributes at most ``validation_per_cell``
        approximately equally spaced cycles.  The subset is defined only by
        sample keys, so every model and checkpoint sees exactly the same
        validation observations.
        """
        dataset = PairedBatteryDataset(
            self.args.data,
            split="vali",
            condition=condition,
            data_root=getattr(self.args, "data_root", None) or DATA_ROOT,
        )
        by_cell: dict[str, list[int]] = {}
        for index, key in enumerate(dataset.keys):
            by_cell.setdefault(key.parent.as_posix(), []).append(index)
        selected: list[int] = []
        per_cell = int(getattr(self.args, "validation_per_cell", 20))
        for cell in sorted(by_cell):
            indices = by_cell[cell]
            count = min(per_cell, len(indices))
            positions = np.unique(np.rint(np.linspace(0, len(indices) - 1, count)).astype(int))
            selected.extend(indices[position] for position in positions)
        return DataLoader(
            Subset(dataset, selected),
            batch_size=int(getattr(self.args, "eval_batch_size", None) or 128),
            shuffle=False,
            num_workers=int(self.args.num_workers),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    @torch.inference_mode()
    def _selection_metrics(self) -> dict[str, Any]:
        """Evaluate the locked five-view reconstruction selection score."""
        self.model.eval()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        condition_rmse: dict[str, float] = {}
        try:
            for condition_index, condition in enumerate(FORMAL_RECONSTRUCTION_CONDITIONS):
                evaluation_seed = int(self.args.seed) + 10_000 + condition_index
                random.seed(evaluation_seed)
                np.random.seed(evaluation_seed)
                torch.manual_seed(evaluation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(evaluation_seed)
                squared_error = 0.0
                point_count = 0
                for current, voltage, clean_voltage, _, _, _ in self._selection_loader(condition):
                    prediction = self._predict_physical(voltage.to(self.device), current.to(self.device))
                    error = (prediction - clean_voltage.to(self.device)).double()
                    squared_error += float(error.square().sum())
                    point_count += error.numel()
                condition_rmse[condition] = float(np.sqrt(squared_error / point_count))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        average = float(np.mean(list(condition_rmse.values())))
        worst = float(max(condition_rmse.values()))
        weight = float(getattr(self.args, "worst_weight", 0.5))
        return {
            "per_condition_rmse": condition_rmse,
            "average_rmse": average,
            "worst_rmse": worst,
            "worst_weight": weight,
            "selection_score": average + weight * worst,
        }

    def _sampling_steps(self) -> int:
        return int(getattr(self.args, "sampling_steps", 1))

    @contextmanager
    def _fixed_evaluation_rng(self, split: str, condition: str):
        """Use one recorded CSDI sampling stream for every model seed."""
        if str(self.args.recon_model).lower() != "csdi":
            yield
            return
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        split_offset = {"vali": 0, "test": 100}.get(split, 200)
        condition_offset = FORMAL_RECONSTRUCTION_CONDITIONS.index(condition)
        seed = int(getattr(self.args, "eval_reconstruction_seed", 3407)) + split_offset + condition_offset
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _sampling_output(self, voltage: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return self.model(V=voltage, I=current, mode="recon_only_sampling", steps=self._sampling_steps())

    def _training_output(self, voltage: torch.Tensor, current: torch.Tensor):
        return self.model(voltage, current, mode="recon_only_train", steps=self._sampling_steps())

    def _predict_physical(self, voltage: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        normalized = self._sampling_output(voltage, current)
        return self._unwrap(self.model)._denormalize_phy(normalized)

    def _training_loss(self, voltage: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        model_name = str(self.args.recon_model).lower()
        output = self._training_output(voltage, current)
        if model_name == "batterydit":
            prediction, target, task_id = output
            pointwise = F.mse_loss(prediction, target)
            gradient = voltage_gradient_loss(
                prediction,
                target,
                voltage_range=float(self.args.v_max - self.args.v_min),
            )
            total = pointwise + BATTERYDIT_GRADIENT_LOSS_WEIGHT * gradient
            return total, {
                "pointwise_mse": float(pointwise.detach()),
                "voltage_gradient_mse": float(gradient.detach()),
                "gradient_loss_weight": BATTERYDIT_GRADIENT_LOSS_WEIGHT,
            }
        if model_name == "csdi":
            prediction, target, target_mask = output
            total = self._unwrap(self.model).compute_loss(prediction, target, target_mask)
            return total, {"pointwise_mse": float(total.detach())}
        if model_name == "transformer":
            prediction, target, task_id = output
            total = self._unwrap(self.model).compute_loss(prediction, target, task_id)
            return total, {"pointwise_mse": float(total.detach())}
        raise RuntimeError("Kalman is eval-only and cannot be trained")

    def _evaluate(
        self,
        split: str,
        condition: str,
        save_examples: bool = False,
        save_trajectories: bool = False,
    ) -> tuple[dict[str, float], list[tuple[np.ndarray, np.ndarray, np.ndarray, str]], dict[str, np.ndarray] | None]:
        loader = self._paired_loader(split, condition)
        self.model.eval()
        sample_rmse: list[np.ndarray] = []
        sample_mae: list[np.ndarray] = []
        sample_cells: list[np.ndarray] = []
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        trajectories: dict[str, list[np.ndarray]] | None = None
        if save_trajectories:
            trajectories = {
                "input_current": [], "input_voltage": [], "clean_voltage": [],
                "reconstruction": [], "soh": [], "rul": [], "keys": [],
            }
        with self._fixed_evaluation_rng(split, condition):
            with torch.no_grad():
                for current, voltage, clean_voltage, soh, rul, keys in loader:
                    current = current.to(self.device)
                    voltage = voltage.to(self.device)
                    clean_voltage = clean_voltage.to(self.device)
                    prediction = self._predict_physical(voltage, current)
                    error = prediction - clean_voltage
                    sample_rmse.append(error.square().mean(dim=(1, 2)).sqrt().cpu().numpy())
                    sample_mae.append(error.abs().mean(dim=(1, 2)).cpu().numpy())
                    sample_cells.append(np.asarray([Path(str(key)).parent.as_posix() for key in keys], dtype=str))
                    if trajectories is not None:
                        trajectories["input_current"].append(current.cpu().numpy())
                        trajectories["input_voltage"].append(voltage.cpu().numpy())
                        trajectories["clean_voltage"].append(clean_voltage.cpu().numpy())
                        trajectories["reconstruction"].append(prediction.cpu().numpy())
                        trajectories["soh"].append(soh.numpy())
                        trajectories["rul"].append(rul.numpy())
                        trajectories["keys"].append(np.asarray(keys, dtype=str))
                    if save_examples and len(examples) < 6:
                        for index in range(min(6 - len(examples), prediction.size(0))):
                            examples.append((
                                voltage[index].cpu().numpy(),
                                clean_voltage[index].cpu().numpy(),
                                prediction[index].cpu().numpy(),
                                str(keys[index]),
                            ))
        rmse = np.concatenate(sample_rmse)
        mae = np.concatenate(sample_mae)
        cells = np.concatenate(sample_cells)
        cell_rmse = np.asarray([rmse[cells == cell].mean() for cell in np.unique(cells)])
        report = {
            "sample_count": int(rmse.size),
            "sample_rmse_mean": float(rmse.mean()),
            "sample_rmse_std": float(rmse.std(ddof=1)) if rmse.size > 1 else 0.0,
            "sample_mae_mean": float(mae.mean()),
            "battery_count": int(cell_rmse.size),
            "battery_balanced_rmse": float(cell_rmse.mean()),
            "battery_balanced_rmse_std": float(cell_rmse.std(ddof=1)) if cell_rmse.size > 1 else 0.0,
        }
        packed = ({key: np.concatenate(value, axis=0) for key, value in trajectories.items()} if trajectories is not None else None)
        return report, examples, packed

    def _save_checkpoint(self, path: Path, epoch: int, record: dict[str, Any]) -> None:
        torch.save({"model": self._unwrap(self.model).state_dict(), "epoch": epoch, "record": record}, path)

    def train(self, setting: str) -> nn.Module:
        if str(self.args.recon_model).lower() == "kalman":
            raise ValueError("Kalman is an eval-only formal baseline; run with --is_training 0")
        checkpoints, results = self._run_paths(setting)
        self._write_config(results)
        model_name = str(self.args.recon_model).lower()
        training = RECONSTRUCTION_TRAINING[model_name]
        trainable = self._unwrap(self.model).recon.parameters() if model_name == "batterydit" else self.model.parameters()
        optimizer = optim.Adam(
            trainable,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        milestones = [int(float(fraction) * int(self.args.epochs)) for fraction in training["scheduler_milestones"]]
        scheduler = (
            optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
            if milestones else None
        )
        train_loader = self._loader("train")
        history: list[dict[str, Any]] = []
        best_score = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, int(self.args.epochs) + 1):
            self.model.train()
            losses: list[float] = []
            components: dict[str, list[float]] = {}
            for current, voltage, _, _ in train_loader:
                current, voltage = current.to(self.device), voltage.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss, values = self._training_loss(voltage, current)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                for key, value in values.items():
                    components.setdefault(key, []).append(value)
            if scheduler is not None:
                scheduler.step()
            record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            record["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            record.update({key: float(np.mean(value)) for key, value in components.items()})
            should_validate = (
                epoch == 1
                or epoch % int(getattr(self.args, "validation_interval", 10)) == 0
                or epoch == int(self.args.epochs)
            )
            if should_validate:
                validation = self._selection_metrics()
                record["validation"] = validation
                if float(validation["selection_score"]) < best_score:
                    best_score = float(validation["selection_score"])
                    best_epoch = epoch
                    stale = 0
                    self._save_checkpoint(checkpoints / "checkpoint.pth", epoch, record)
                else:
                    stale += 1
            history.append(record)
            self._save_checkpoint(checkpoints / "last.pth", epoch, record)
            (results / "history.json").write_text(json.dumps(history, indent=2))
            print(f"epoch {epoch}: train={record['train_loss']:.6g}")
            if should_validate and stale >= int(getattr(self.args, "patience", 10)):
                print(
                    f"early stopping after epoch {epoch}; best epoch={best_epoch}, "
                    f"selection score={best_score:.6g}"
                )
                break
        (results / "selection.json").write_text(json.dumps({
            "rule": "mean waveform RMSE + worst_weight * worst-condition RMSE",
            "conditions": list(FORMAL_RECONSTRUCTION_CONDITIONS),
            "validation_per_cell": int(getattr(self.args, "validation_per_cell", 20)),
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "test_opened": False,
        }, indent=2, sort_keys=True))
        return self.model

    def vali(self, setting: str) -> dict[str, float]:
        _, results = self._run_paths(setting)
        reports = {}
        for condition in FORMAL_RECONSTRUCTION_CONDITIONS:
            report, _, _ = self._evaluate("vali", condition)
            reports[condition] = report
        (results / "validation_metrics.json").write_text(json.dumps({"conditions": reports}, indent=2, sort_keys=True))
        return reports

    def _load_best_checkpoint(self, checkpoints: Path) -> None:
        requested = str(getattr(self.args, "recn_chkpt", "")).strip()
        path = Path(requested) if requested else checkpoints / "checkpoint.pth"
        if not path.is_file():
            if str(self.args.recon_model).lower() == "kalman":
                return
            raise FileNotFoundError(f"reconstruction checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        load_reconstruction_state_dict(self._unwrap(self.model), state, strict=True)

    def test(self, setting: str) -> dict[str, dict[str, float]]:
        checkpoints, results = self._run_paths(setting)
        self._write_config(results)
        self._load_best_checkpoint(checkpoints)
        reports: dict[str, dict[str, float]] = {}
        for condition in FORMAL_RECONSTRUCTION_CONDITIONS:
            report, examples, trajectories = self._evaluate(
                "test", condition, save_examples=True,
                save_trajectories=bool(getattr(self.args, "save_reconstructions", False)),
            )
            reports[condition] = report
            if examples:
                np.savez_compressed(
                    results / f"examples_{condition}.npz",
                    input=np.stack([item[0] for item in examples]),
                    target=np.stack([item[1] for item in examples]),
                    reconstruction=np.stack([item[2] for item in examples]),
                    keys=np.asarray([item[3] for item in examples]),
                )
            if trajectories is not None:
                np.savez_compressed(results / f"reconstructions_{condition}.npz", **trajectories)
        (results / "test_metrics.json").write_text(json.dumps({"protocol": PROTOCOL_NAME, "conditions": reports}, indent=2, sort_keys=True))
        return reports
