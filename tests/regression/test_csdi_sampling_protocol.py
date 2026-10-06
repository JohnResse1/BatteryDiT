"""Regression contract for the formal CSDI reverse-diffusion schedule."""
from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from models.recon_model.Baseline_CSDI_official import Model


def main() -> None:
    args = SimpleNamespace(
        diffusion_steps=7,
        csdi_official_time_dim=8,
        csdi_official_feature_dim=4,
        csdi_n_samples=1,
        csdi_sample_clip=10.0,
        csdi_layers=1,
        csdi_channels=8,
        csdi_official_heads=1,
        csdi_diffusion_dim=8,
        i_min=0.0,
        i_max=6.2,
        v_min=2.0,
        v_max=3.6,
    )
    model = Model(args)
    used_steps: list[int] = []

    def record_schedule(self, observed_data, cond_mask, side_info, sampling_steps):
        del self, cond_mask, side_info
        used_steps.append(sampling_steps)
        return observed_data

    model._impute_once = MethodType(record_schedule, model)
    voltage = torch.tensor([[[3.4], [0.0], [3.0], [2.8]]])
    current = torch.ones_like(voltage)
    model(voltage, current, mode="recon_only_sampling", steps=1)
    assert used_steps == [args.diffusion_steps]
    print("CSDI full reverse-diffusion protocol passed")


if __name__ == "__main__":
    main()
