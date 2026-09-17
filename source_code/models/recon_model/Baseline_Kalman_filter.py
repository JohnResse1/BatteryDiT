import torch
import torch.nn as nn

class Model(nn.Module):
    """
    工业级鲁棒一维卡尔曼滤波器 (Robust 1D Kalman Filter Baseline)
    - 完美兼容现有的测试接口 (V, I, mode, steps)。
    - 无需训练：前向传播直接进行因果滤波推断。
    - 带有“新息门控 (Innovation Gating)”：遇到超过物理阈值的跳变，直接熔断并抛弃。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 卡尔曼滤波核心物理超参数
        # Q: 过程噪声协方差 (越小代表越相信电池电压是平滑连续的热力学流形)
        self.Q = getattr(args, "kf_q", 1e-5)
        # R: 测量噪声协方差 (代表传感器的基础高频底噪)
        self.R = getattr(args, "kf_r", 1e-2)
        # 异常门控阈值 (单位：V)。任何单步超过 100mV 的跳变都会被直接视为硬件尖峰并抛弃！
        self.outlier_threshold = getattr(args, "kf_threshold", 0.1)

        # 保持接口兼容所需
        self.pred = nn.Identity()

    def get_learned_bounds(self):
        # 仅为兼容接口
        return torch.tensor([0, 0, 0])

    def _normalize_phy(self, v_phys: torch.Tensor) -> torch.Tensor:
        """与其它重构模型保持统一接口；Kalman baseline 默认直接在原始电压域工作。"""
        return v_phys

    def _denormalize_phy(self, v_norm: torch.Tensor) -> torch.Tensor:
        """与其它重构模型保持统一接口；Kalman baseline 默认直接在原始电压域工作。"""
        return v_norm

    def compute_loss(self, v_pred: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        # 仅为兼容接口
        loss = torch.mean((v_pred - target) ** 2)
        return loss

    def _corrupt_inputs(self, V: torch.Tensor, I: torch.Tensor, base_mask: torch.Tensor):
        # 仅为兼容训练模式接口，保证不报错
        return V.clone(), I.clone(), base_mask.clone(), 0

    def forward(self, V: torch.Tensor, I: torch.Tensor, mode: str = 'recon_only_train', steps: int = 1):
        """
        统一的前向传播接口。
        注意：KF 是在【绝对物理电压】上工作的，不需要像扩散模型那样映射到 [-1, 1]。
        """
        if V.dim() == 2:
            V = V.unsqueeze(-1)
        if I.dim() == 2:
            I = I.unsqueeze(-1)

        device = V.device
        B, L, C = V.shape

        # 获取真实的有效数据掩码 (剔除掉因为缺失或 padding 导致的 0 值)
        base_mask = ((torch.abs(V) > 1e-6) & ~torch.isnan(V)).float()

        if mode == 'recon_only_train':
            # KF 无需训练！如果误触训练代码，返回全零 dummy 梯度以防崩溃
            v_raw, i_raw, cond_mask, task_id = self._corrupt_inputs(V, I, base_mask)
            eps_pred = torch.zeros_like(V, requires_grad=True)
            eps = torch.zeros_like(V)
            return eps_pred, eps, task_id

        if mode == 'recon_only_sampling':
            V_noisy = V.clone()
            mask = base_mask.clone()

            V_filtered = torch.zeros_like(V_noisy)

            # 【修复 1】：安全初始化。不再硬编码 3.6V，以防止数据已经被外部 Normalize。
            # 找到每个 batch 中第一个有效的值作为初始估计。若全空，则设为 0。
            x_est = torch.zeros((B, C), device=device)
            for b in range(B):
                valid_idx = torch.nonzero(mask[b, :, 0]).squeeze()
                if valid_idx.numel() > 0:
                    first_valid = valid_idx[0] if valid_idx.dim() == 1 else valid_idx
                    x_est[b] = V_noisy[b, first_valid]

            P_est = torch.ones_like(x_est) * self.R

            # 【新增】：断连时间计数器，用于解决“死锁”
            time_since_last_valid = torch.zeros((B, C), device=device)

            # 2. 逐时间步递推滤波
            for k in range(L):
                x_pred = x_est
                P_pred = P_est + self.Q

                z = V_noisy[:, k, :]
                m = mask[:, k, :]

                innovation = torch.abs(z - x_pred)

                # 【修复 2】：带有“重新锚定”的异常门控
                # 只有当偏差大于阈值，【并且】我们最近刚看到过有效数据（断连 < 5步）时，才判定为尖峰！
                # 如果断连超过了 5 步，说明这是恢复通讯后的第一个值，强制信任它以防死锁！
                is_outlier = (innovation > self.outlier_threshold) & (time_since_last_valid < 5)
                is_outlier = is_outlier.float()

                effective_mask = m * (1.0 - is_outlier)

                # 更新断连计数器：如果接受了数据，计数器清零；否则加 1
                time_since_last_valid = torch.where(effective_mask > 0, torch.zeros_like(time_since_last_valid), time_since_last_valid + 1)

                K_gain = P_pred / (P_pred + self.R)

                # 更新步
                x_est = x_pred + effective_mask * K_gain * (z - x_pred)
                P_est = (1.0 - effective_mask * K_gain) * P_pred

                V_filtered[:, k, :] = x_est

            return V_filtered

        raise ValueError(f"Unsupported mode: {mode}")
