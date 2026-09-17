import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.recon_model import dit_backbone
from models.checkpoint_loading import load_reconstruction_state_dict
from .po_factory import create_physics_operator
from utils.tools import generate_hybrid_mask, inject_compound_noise_torch, inject_destructive_real_noise_torch


class PhysicsConsistencyMaskStrategy:
    """Fixed-rule physics-consistency mask.

    Cross-modal V/I inconsistency is mapped to an observation-confidence
    weight through a Kalman-gain-shaped function.  This mapping is a fixed
    rule in the current hard-mask pipeline, not a learned mask network.
    """
    def __init__(self, device, sigma_po=1.0, sigma_sensor_base=0.01, gamma=1.5, epsilon=0.02, explosion_factor=25.0, smooth_kernel=5):
        self.device = device
        self.sigma_po = sigma_po
        self.sigma_sensor_base = sigma_sensor_base
        self.gamma = gamma
        self.epsilon = epsilon
        self.explosion_factor = explosion_factor
        self.smooth_kernel = smooth_kernel


    def to_hard_mask(self, soft_mask, tau=0.5):
        return (soft_mask > tau).float()

    # Historical typo retained for old analysis notebooks/checkpoints.
    def to_har_mask(self, soft_mask, tau=0.5):
        return self.to_hard_mask(soft_mask, tau=tau)


    # V, I relative differences
    def _compute_rel_diffs(self, V, I, is_valid):
        if V.dim() == 2: V, I = V.unsqueeze(-1), I.unsqueeze(-1)
        B, L, C = V.shape
        zeros = torch.zeros(B, 1, C, device=self.device)

        v_diff_abs = torch.cat([zeros, torch.abs(V[:, 1:, :] - V[:, :-1, :])], dim=1)
        i_diff_abs = torch.cat([zeros, torch.abs(I[:, 1:, :] - I[:, :-1, :])], dim=1)

        valid_diff_mask = torch.cat([zeros, is_valid[:, 1:, :] * is_valid[:, :-1, :]], dim=1)
        v_diff_abs = v_diff_abs * valid_diff_mask
        i_diff_abs = i_diff_abs * valid_diff_mask

        # get relative differences by normalizing with mean absolute values, with clamping to avoid extreme ratios
        v_mean = torch.sum(torch.abs(V), dim=1, keepdim=True) / (torch.sum(is_valid, dim=1, keepdim=True) + 1e-6)
        i_mean = torch.sum(torch.abs(I), dim=1, keepdim=True) / (torch.sum(is_valid, dim=1, keepdim=True) + 1e-6)
        v_scale = torch.clamp(v_mean, min=2.0, max=5.0)
        i_scale = torch.clamp(i_mean, min=0.5, max=20.0)
        v_diff_rel = v_diff_abs / v_scale
        i_diff_rel = i_diff_abs / i_scale
        return v_diff_rel, i_diff_rel, valid_diff_mask

    # 1d average pooling for smoothing the discrepancy signal, with edge padding to maintain sequence length
    def _smooth(self, x):
        if self.smooth_kernel <= 1:
            return x
        x_in = x.permute(0, 2, 1)
        pad = self.smooth_kernel // 2
        x_smooth = F.avg_pool1d(x_in, kernel_size=self.smooth_kernel, stride=1, padding=pad)
        return x_smooth.permute(0, 2, 1)

    def get_mask(self, V_obs_raw, I_obs_raw, min_conf, weak_prior, metadata_mask):
        # 1) 有效性检测
        is_valid = ((torch.abs(V_obs_raw) > 1e-6) & ~torch.isnan(V_obs_raw)).float()

        # 2) 无量纲相对跳变与拓扑违和度
        v_diff_rel, i_diff_rel, valid_diff_mask = self._compute_rel_diffs(V_obs_raw, I_obs_raw, is_valid)
        discrepancy = F.relu(v_diff_rel - self.gamma * i_diff_rel - self.epsilon) * valid_diff_mask
        discrepancy = self._smooth(discrepancy)

        # 3) 观测噪声膨胀与卡尔曼增益
        sigma_sensor_sq = self.sigma_sensor_base + torch.expm1(self.explosion_factor * discrepancy)
        kalman_mask = self.sigma_po / (self.sigma_po + sigma_sensor_sq)

        # 4) 下界约束 + 稀疏先验
        # torch.clamp requires max to match the tensor/number type of min_conf
        max_conf = torch.ones_like(kalman_mask)
        physics_conf = torch.clamp(kalman_mask, min=min_conf, max=max_conf) * is_valid
        weak_prior_tensor = weak_prior * torch.ones_like(physics_conf)
        # sparse_conf = self.get_sparsity_confidence(metadata_mask)
        # anchor_conf = torch.min(physics_conf, sparse_conf)
        # weak_prior_tensor = weak_prior * torch.ones_like(anchor_conf)

        # 5) 组装输出：观测点用卡尔曼增益，缺失点回退 weak_prior
        final_mask = torch.where(
            metadata_mask > 0.5,
            physics_conf,
            weak_prior_tensor
        )
        return final_mask


# Backward-compatible import used by older notebooks.
IntegratedMaskStrategy = PhysicsConsistencyMaskStrategy


class Model(nn.Module):
    """

    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.recon = dit_backbone.Model(args)

        # =======================================
        # Physics Operator Factory with Optional Custom OCV
        # =======================================
        ocv_soc = None
        ocv_voltage = None
        ocv_path = getattr(self.args, "ocv_table_path", "")
        if ocv_path:
            try:
                with np.load(ocv_path) as ocv_npz:
                    if "soc" not in ocv_npz or "voltage" not in ocv_npz:
                        raise ValueError("OCV table must contain 'soc' and 'voltage' arrays")
                    ocv_soc = ocv_npz["soc"]
                    ocv_voltage = ocv_npz["voltage"]
                print(f"[Physics Operator] Loaded custom OCV table from {ocv_path} (len={len(ocv_soc)})")
            except Exception as error:
                print(f"[Physics Operator] Failed to load OCV table from {ocv_path}, fallback to default OCV. Error: {error}")

        self.po = create_physics_operator(
            self.args,
            current_scale=1.0,
            ocv_soc=ocv_soc,
            ocv_voltage=ocv_voltage,
            prefix="[Physics Operator]",
        )

        # ======================================
        # Physics-consistency mask bounds.  The legacy raw_* state names are
        # intentionally retained so every existing checkpoint loads unchanged.
        # With the current hard threshold these values act as fixed settings.
        # ======================================
        self.raw_min_conf = nn.Parameter(torch.tensor(0.0))
        self.raw_weak_prior = nn.Parameter(torch.tensor(-2.0))
        # Legacy unused key retained strictly for checkpoint compatibility.
        self.raw_weak_prior_max = nn.Parameter(torch.tensor(0.0))
        self.mask_strategy = PhysicsConsistencyMaskStrategy(device=None)

        # hard mask threshold (for analysis only, not used in training)
        self.use_hard_mask = True # false to use soft mask during training, true to apply hard thresholding for evaluation/analysis
        self.hard_mask_threshold = 0.5

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load historical reconstruction checkpoints without their unused MLP."""
        return load_reconstruction_state_dict(
            self, state_dict, strict=strict, assign=assign
        )


    def _normalize_phy(self, v_phys):
        return (v_phys - self.args.v_min) / (self.args.v_max - self.args.v_min)

    def _denormalize_phy(self, v_norm):
        return v_norm * (self.args.v_max - self.args.v_min) + self.args.v_min

    def get_mask_bounds(self):
        """Map the legacy raw state values to fixed mask bounds."""
        min_conf = 0.4 * torch.sigmoid(self.raw_min_conf) + 0.3
        weak_prior = 0.4 * torch.sigmoid(self.raw_weak_prior)
        return min_conf, weak_prior

    # Backward-compatible API for historical experiment and plotting code.
    def get_learned_bounds(self):
        return self.get_mask_bounds()

    def _sample_from_choices(self, choices):
        idx = torch.randint(0, len(choices), (1,)).item()
        return choices[idx]

    def _robust_linear_interpolate(self, I, mask):
        """向量化抗噪线性插值"""
        B, L, C = I.shape
        device = I.device
        indices = torch.arange(L, device=device).view(1, L, 1).expand(B, L, C)
        is_valid = (mask > 0.5)

        left_indices = torch.where(is_valid, indices, torch.full_like(indices, -1))
        left_anchors_idx, _ = torch.cummax(left_indices, dim=1)
        right_indices = torch.where(is_valid, indices, torch.full_like(indices, L))
        right_anchors_idx, _ = torch.cummin(right_indices.flip(1), dim=1)
        right_anchors_idx = right_anchors_idx.flip(1)

        left_anchors_idx[left_anchors_idx == -1] = right_anchors_idx[left_anchors_idx == -1]
        right_anchors_idx[right_anchors_idx == L] = left_anchors_idx[right_anchors_idx == L]
        left_anchors_idx = torch.clamp(left_anchors_idx, 0, L-1)
        right_anchors_idx = torch.clamp(right_anchors_idx, 0, L-1)

        val_left = torch.gather(I, 1, left_anchors_idx)
        val_right = torch.gather(I, 1, right_anchors_idx)

        dist_total = (right_anchors_idx - left_anchors_idx).float()
        dist_total[dist_total == 0] = 1.0
        dist_curr = (indices - left_anchors_idx).float()
        alpha = dist_curr / dist_total

        I_interpolated = val_left + alpha * (val_right - val_left)
        return I * mask + I_interpolated * (1 - mask)

    def compute_loss(self, v_pred, target_velocity, task_id):
        return torch.mean((v_pred - target_velocity) ** 2)

    def forward(self, V, I, mode='all', steps=1):
        batch_size, seq_len, _ = V.shape
        device = V.device
        self.mask_strategy.device = device

        if mode == 'recon_only_train':
            x_1_target = self._normalize_phy(V)
            p = torch.rand(1).item()

            v_raw = V.clone()
            i_raw = I.clone()
            metadata_mask = torch.ones(batch_size, seq_len, 1, device=device)

            # --- 元任务场景分支 ---
            if p < 0.2:
                task_id = 3 # Clean
            elif p < 0.5:
                task_id = 1 # Destructive
                p_real = torch.rand(1).item()
                if p_real < 0.5:
                    v_raw, i_raw = inject_destructive_real_noise_torch(V, I)
                else:
                    noise_lvl = torch.rand(1).item() * 0.06 + 0.02
                    v_raw, i_raw = inject_compound_noise_torch(V, I, noise_level_base=noise_lvl)
            elif p < 0.7:
                task_id = 2 # Downsample
                down_ratio = torch.randint(2, 10, (1,)).item()
                metadata_mask = torch.zeros_like(metadata_mask)
                metadata_mask[:, ::down_ratio, :] = 1.0
                v_raw = V * metadata_mask
                i_raw = I * metadata_mask
            else:
                task_id = 0 # Missing + Noise (segment)
                mask_ratio = self._sample_from_choices([0.2, 0.4, 0.6, 0.8])
                metadata_mask = generate_hybrid_mask(batch_size, seq_len, 1, mask_ratio=mask_ratio).float().to(device)
                noise_lvl = self._sample_from_choices([0.02, 0.04, 0.06, 0.08])
                v_noisy, i_noisy = inject_compound_noise_torch(V, I, noise_level_base=noise_lvl)
                v_raw = v_noisy * metadata_mask
                i_raw = i_noisy * metadata_mask

            # --- 物理约束与融合起跑线 ---
            # get bounds
            min_conf, weak_prior = self.get_mask_bounds()
            # get masks
            mask = self.mask_strategy.get_mask(v_raw, i_raw, min_conf, weak_prior, metadata_mask)

            if self.use_hard_mask:
                mask = self.mask_strategy.to_har_mask(mask, tau=self.hard_mask_threshold)

            # get interpolation results for the sensor input under segment missing
            v_interp = self._robust_linear_interpolate(v_raw, metadata_mask)
            x_sensor = self._normalize_phy(v_interp)
            i_interp = self._robust_linear_interpolate(i_raw, metadata_mask)
            # get PO input from interpolated current
            x_po_input = self._normalize_phy(self.po(i_interp))
            # Base flow assembly: mask * sensor + (1-mask) * physics
            x_base = mask * x_sensor + (1 - mask) * x_po_input

            target_velocity = x_1_target - x_base

            # 构造连续路径 X_t
            t = torch.rand(batch_size, device=device)
            t_expand = t.view(batch_size, 1, 1)
            x_t = t_expand * x_1_target + (1 - t_expand) * x_base

            # DiT 仅接收纯粹连续的电压流形 (2通道)
            model_input = torch.cat([x_t, x_po_input], dim=-1)
            predicted_velocity = self.recon(model_input, t)

            return predicted_velocity, target_velocity, task_id

        elif mode == 'recon_only_sampling':
            # 1. 原始有效性检测（V/I 联合）：避免 I 中 NaN 在插值阶段被当作“有效观测”泄漏
            v_valid = ((torch.abs(V) > 1e-6) & ~torch.isnan(V))
            i_valid = ~torch.isnan(I)
            raw_metadata_mask = (v_valid & i_valid).float()

            V_safe = torch.nan_to_num(V, nan=0.0) * raw_metadata_mask
            I_safe = torch.nan_to_num(I, nan=0.0) * raw_metadata_mask
            # ==========================================================

            # Fixed physics-consistency mask bounds.
            min_conf, weak_prior = self.get_mask_bounds()

            # 2. 计算软掩码：喂进去的是去毒后的 V/I
            mask = self.mask_strategy.get_mask(V_safe, I_safe, min_conf, weak_prior, raw_metadata_mask)
            if self.use_hard_mask:
                mask = self.mask_strategy.to_har_mask(mask, tau=self.hard_mask_threshold)
            # 3. 构建纯净起跑线：
            # 此时对于原来的尖峰点，V_safe 是 0，它会被这段插值代码完美地用前后的正常值“凌空架桥”铺满！
            v_interp = self._robust_linear_interpolate(V_safe, raw_metadata_mask)
            x_sensor = self._normalize_phy(v_interp)

            i_interp = self._robust_linear_interpolate(I_safe, raw_metadata_mask)
            x_po_input = self._normalize_phy(self.po(i_interp))

            # 【起跑线组装】：
            # 此时尖峰点的 mask 即使泄漏了 weak_prior (如 0.15)，
            # 它乘上的也是 x_sensor (经过干净插值后的完美数据)，而不是 0V 的毒药！
            x_base = mask * x_sensor + (1 - mask) * x_po_input

            # 4. DiT 连续流推断
            d_t = 1.0 / steps
            x_t = x_base.clone()

            for i in range(steps):
                t_scaler = (torch.ones(batch_size, device=device) * i) / steps

                model_input = torch.cat([x_t, x_po_input], dim=-1)
                velocity = self.recon(model_input, t_scaler)
                x_t = x_t + velocity * d_t

            return x_t
