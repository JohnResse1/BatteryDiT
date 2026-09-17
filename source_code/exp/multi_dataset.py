"""Run the formal multi-dataset reconstruction/TRB protocol."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.dataset_physics import dataset_physics

from .downstream_prediction import Exp_Predict_Downstream
from .reconstruction import Exp_Reconstruction, SOURCE_CODE_ROOT


PROTOCOL_PATH = SOURCE_CODE_ROOT / "configs/formal_protocols/multi_dataset_reconstruction.json"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Exp_MultiDataset:
    """Execute the retained multi-dataset protocol without duplicating runners."""

    def __init__(self, args: Any):
        self.args = args
        self.protocol = json.loads(PROTOCOL_PATH.read_text())

    def _selected(self, key: str, configured: tuple | list) -> list:
        requested = list(getattr(self.args, key, []) or [])
        if not requested or requested == ["all"]:
            return list(configured)
        unknown = sorted(set(requested) - set(configured))
        if unknown:
            raise ValueError(f"{key} contains values outside the formal protocol: {unknown}")
        return requested

    def _args_for(self, dataset: str, seed: int, task_name: str) -> Any:
        args = copy.copy(self.args)
        args.task_name = task_name
        args.data = dataset
        args.seed = seed
        args.model_id = f"seed{seed}"
        args.itr = 1
        args.recon_model = "batterydit"
        args._result_group = "multi_dataset"
        args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
        physics = dataset_physics(dataset)
        args.po_type, args.v_min, args.v_max = physics.po_type, physics.v_min, physics.v_max
        args.ocv_table_path = f"models/ocv/{dataset}.npz"
        return args

    @staticmethod
    def _setting(task_name: str, dataset: str, seed: int) -> str:
        return f"{task_name}_seed{seed}_batterydit_{dataset}_0"

    def _run_reconstruction(self, dataset: str, seed: int) -> None:
        args = self._args_for(dataset, seed, "Reconstruction")
        spec = self.protocol["reconstruction"]
        args.epochs, args.batch_size = int(spec["epochs"]), int(spec["batch_size"])
        args.learning_rate, args.weight_decay = float(spec["learning_rate"]), float(spec["weight_decay"])
        args.sampling_steps = int(spec["sampling_steps"])
        _set_seed(seed)
        setting = self._setting(args.task_name, dataset, seed)
        experiment = Exp_Reconstruction(args)
        experiment.train(setting); experiment.vali(setting); experiment.test(setting)

    def _run_downstream(self, dataset: str, seed: int) -> None:
        args = self._args_for(dataset, seed, "PredictDownstream")
        spec = self.protocol["downstream"]
        args.epochs, args.batch_size = int(spec["epochs"]), int(spec["batch_size"])
        args.learning_rate, args.weight_decay = float(spec["learning_rate"]), float(spec["weight_decay"])
        args.validation_interval = int(spec["validation_interval"])
        reconstruction_setting = self._setting("Reconstruction", dataset, seed)
        args.recn_chkpt = str(
            SOURCE_CODE_ROOT / args.checkpoints / "Reconstruction" / dataset
            / reconstruction_setting / "checkpoint.pth"
        )
        if not Path(args.recn_chkpt).is_file():
            raise FileNotFoundError(f"matching reconstruction checkpoint is required before TRB: {args.recn_chkpt}")
        _set_seed(seed)
        setting = self._setting(args.task_name, dataset, seed)
        experiment = Exp_Predict_Downstream(args)
        experiment.train(setting); experiment.vali(setting); experiment.test(setting)

    def train(self, setting: str) -> None:
        if not bool(self.args.is_training):
            raise ValueError("MultiDataset performs training; use --is_training 1")
        datasets = self._selected("multi_datasets", self.protocol["datasets"])
        seeds = self._selected("multi_seeds", self.protocol["seeds"])
        stage = str(self.args.multi_stage)
        for dataset in datasets:
            for seed in seeds:
                if stage in {"reconstruction", "all"}:
                    self._run_reconstruction(dataset, int(seed))
                if stage in {"downstream", "all"}:
                    self._run_downstream(dataset, int(seed))

    def vali(self, setting: str) -> None:
        return None

    def test(self, setting: str) -> None:
        return None
