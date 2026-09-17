"""Ablation model aligned with the current BatteryDiT training pipeline.

The ``all`` branch is kept numerically identical to ``BatteryDiT.Model``.
Every branch uses
the same two-channel DiT backbone, while an absent auxiliary signal is
represented by zeros rather than by the physical value ``1`` (3.6 V after
denormalization in the MIT setup).
"""

import torch

from models import BatteryDiT
from utils.tools import (
    generate_hybrid_mask,
    inject_compound_noise_torch,
    inject_destructive_real_noise_torch,
)


class Model(BatteryDiT.Model):
    """Capacity-matched BatteryDiT component ablations.

    Modes
    -----
    all
        Current BatteryDiT: confidence-mask fusion, PO as channel 2.
    sensor
        Interpolated sensor only, zero auxiliary channel.
    sensor+po
        Equal sensor/PO fusion without the confidence mask, PO as channel 2.
    sensor+mask
        Sensor flow base with the confidence map as channel 2.  This is the
        confidence-only branch; no PO values are exposed to the DiT.
    """

    VALID_ABLATIONS = {"all", "sensor", "sensor+po", "sensor+mask"}

    @classmethod
    def _validate_ablation_mode(cls, ablation_mode):
        mode = ablation_mode or "all"
        if mode not in cls.VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation_mode={mode!r}; choose from "
                f"{sorted(cls.VALID_ABLATIONS)}"
            )
        return mode

    @staticmethod
    def _assemble_branch(mode, x_sensor, x_po, confidence):
        """Return ``(x_base, auxiliary_channel)`` for a fixed 2-channel DiT."""
        if mode == "all":
            return confidence * x_sensor + (1.0 - confidence) * x_po, x_po
        if mode == "sensor+po":
            return 0.5 * x_sensor + 0.5 * x_po, x_po
        if mode == "sensor+mask":
            return x_sensor, confidence
        return x_sensor, torch.zeros_like(x_sensor)

    def _make_components(self, V, I, metadata_mask, ablation_mode):
        min_conf, weak_prior = self.get_mask_bounds()
        confidence = self.mask_strategy.get_mask(
            V, I, min_conf, weak_prior, metadata_mask
        )
        if self.use_hard_mask:
            confidence = self.mask_strategy.to_har_mask(
                confidence, tau=self.hard_mask_threshold
            )

        v_interp = self._robust_linear_interpolate(V, metadata_mask)
        i_interp = self._robust_linear_interpolate(I, metadata_mask)
        x_sensor = self._normalize_phy(v_interp)
        x_po = self._normalize_phy(self.po(i_interp))
        return self._assemble_branch(
            ablation_mode, x_sensor, x_po, confidence
        )

    def forward(
        self,
        V,
        I,
        mode="all",
        steps=1,
        ablation_mode="all",
    ):
        ablation_mode = self._validate_ablation_mode(ablation_mode)
        batch_size, seq_len, _ = V.shape
        device = V.device
        self.mask_strategy.device = device

        if mode == "recon_only_train":
            x_1_target = self._normalize_phy(V)
            p = torch.rand(1).item()

            v_raw = V.clone()
            i_raw = I.clone()
            metadata_mask = torch.ones(
                batch_size, seq_len, 1, device=device
            )

            # Keep the current BatteryDiT meta-task distribution unchanged.
            if p < 0.2:
                task_id = 3  # clean
            elif p < 0.5:
                task_id = 1  # destructive
                p_real = torch.rand(1).item()
                if p_real < 0.5:
                    v_raw, i_raw = inject_destructive_real_noise_torch(V, I)
                else:
                    noise_lvl = torch.rand(1).item() * 0.06 + 0.02
                    v_raw, i_raw = inject_compound_noise_torch(
                        V, I, noise_level_base=noise_lvl
                    )
            elif p < 0.7:
                task_id = 2  # downsample
                down_ratio = torch.randint(2, 10, (1,)).item()
                metadata_mask = torch.zeros_like(metadata_mask)
                metadata_mask[:, ::down_ratio, :] = 1.0
                v_raw = V * metadata_mask
                i_raw = I * metadata_mask
            else:
                task_id = 0  # hybrid missing + noise
                mask_ratio = self._sample_from_choices([0.2, 0.4, 0.6, 0.8])
                metadata_mask = generate_hybrid_mask(
                    batch_size,
                    seq_len,
                    1,
                    mask_ratio=mask_ratio,
                    device=device,
                ).float()
                noise_lvl = self._sample_from_choices([0.02, 0.04, 0.06, 0.08])
                v_noisy, i_noisy = inject_compound_noise_torch(
                    V, I, noise_level_base=noise_lvl
                )
                v_raw = v_noisy * metadata_mask
                i_raw = i_noisy * metadata_mask

            x_base, auxiliary = self._make_components(
                v_raw, i_raw, metadata_mask, ablation_mode
            )
            target_velocity = x_1_target - x_base

            t = torch.rand(batch_size, device=device)
            t_expand = t.view(batch_size, 1, 1)
            x_t = t_expand * x_1_target + (1.0 - t_expand) * x_base
            predicted_velocity = self.recon(
                torch.cat([x_t, auxiliary], dim=-1), t
            )
            return predicted_velocity, target_velocity, task_id

        if mode == "recon_only_sampling":
            v_valid = (torch.abs(V) > 1e-6) & ~torch.isnan(V)
            i_valid = ~torch.isnan(I)
            metadata_mask = (v_valid & i_valid).float()
            v_safe = torch.nan_to_num(V, nan=0.0) * metadata_mask
            i_safe = torch.nan_to_num(I, nan=0.0) * metadata_mask

            x_base, auxiliary = self._make_components(
                v_safe, i_safe, metadata_mask, ablation_mode
            )
            if steps < 1:
                raise ValueError(f"steps must be >= 1, got {steps}")
            x_t = x_base.clone()
            d_t = 1.0 / steps
            for step in range(steps):
                t = torch.full(
                    (batch_size,), step / steps, device=device
                )
                velocity = self.recon(
                    torch.cat([x_t, auxiliary], dim=-1), t
                )
                x_t = x_t + velocity * d_t
            return x_t

        raise NotImplementedError(f"Unsupported mode: {mode}")
