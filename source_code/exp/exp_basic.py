"""Shared base class for formal experiment runners."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.BatteryDiT import Model as BatteryDiT
from models.recon_model.Baseline_CSDI_official import Model as CSDIOfficial
from models.recon_model.Baseline_Kalman_filter import Model as KalmanFilter
from models.recon_model.Baseline_Transformer import Model as PlainTransformer


class Exp_Basic:
    """Build a reconstruction model and provide the common experiment API.

    Subclasses implement ``train``, ``vali``, and ``test``.  Keeping this
    small base class preserves the original project entry style while keeping
    model selection in exactly one place.
    """

    model_dict: dict[str, type[nn.Module]] = {
        "batterydit": BatteryDiT,
        "csdi": CSDIOfficial,
        "transformer": PlainTransformer,
        "kalman": KalmanFilter,
    }

    def __init__(self, args: Any):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self) -> torch.device:
        if bool(getattr(self.args, "use_gpu", False)) and torch.cuda.is_available():
            return torch.device(f"cuda:{int(getattr(self.args, 'gpu', 0))}")
        return torch.device("cpu")

    def _build_model(self) -> nn.Module:
        name = str(getattr(self.args, "recon_model", "batterydit")).lower()
        try:
            return self.model_dict[name](self.args)
        except KeyError as error:
            choices = ", ".join(self.model_dict)
            raise ValueError(f"unknown formal reconstruction model {name!r}; choose {choices}") from error

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, nn.DataParallel) else model

    def train(self, setting: str):
        raise NotImplementedError

    def vali(self, setting: str):
        raise NotImplementedError

    def test(self, setting: str):
        raise NotImplementedError
