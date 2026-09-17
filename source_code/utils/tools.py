import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # type1是快速收敛策略
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}  # 默认使用这种随epoch下降的学习率
    # type2是某种通过测试得到大概结果
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    # type3是单一不变学习率,这个学习率也较大
    elif args.lradj == 'type3':
        lr_adjust = {1: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=100, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    # early stopping只接受一个val_loss就可以了，只追求val下降就好。只是val不下降的时候模型不更新
    def __call__(self, val_loss, model, path):
        score = -val_loss  # loss是正值，loss越小，越接近0，反而负值越大，所以score就越大
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:  # 如果score小，说明loss变大，则继续count
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # Early stopping
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 如果val_loss降低
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None,label_x = None,label_y =None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    # plt.ylim(0,0.01)
    plt.plot(true, linewidth=2)
    if preds is not None:
        plt.plot(preds, linewidth=2)
    if label_x is not None:
        plt.xlabel(label_x)
    if label_y is not None:
        plt.ylabel(label_y)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_true_pred_input(true, preds, input_data, name='./pic/test_3d.pdf',):
    plt.figure()
    plt.plot(true, label='True', linewidth=2)
    plt.plot(preds, label='Predicted', linewidth=2)
    plt.plot(input_data, label='Input', linewidth=2)
    plt.title(name)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')



def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def visualize_losses(total_train_loss, total_test1_loss, save_path):
    epochs = range(1, len(total_train_loss) + 1)
    plt.figure()
    plt.plot(epochs, total_train_loss, label='Train Loss')
    plt.plot(epochs, total_test1_loss, label='Test1 Loss')

    plt.title('Training, and Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(save_path)
    plt.show()

def generate_hybrid_mask(batch_size, seq_length, feature_num, mask_ratio=0.6, device=None   ):
    """
    Generate a hybrid mask to simulate missing data patterns:
    Random points missing and random contiguous blocks missing.
    This is for training the models.
    """
    batch_size = int(batch_size)
    seq_length = int(seq_length)
    feature_num = int(feature_num)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # compute target_masked_count by mask_ratio
    mask = torch.ones(batch_size, seq_length, feature_num, dtype=torch.bool, device=device)
    total_elements = seq_length * feature_num
    target_masked_count = int(total_elements * mask_ratio)

    # generate mask for each sample in the batch
    for i in range(batch_size):
        strategy = torch.rand(1).item()

        # --- Block Masking ---
        # 70% probability to generate *mask_ratio* length block missing
        if strategy > 0.3:
            min_len = int(seq_length * 0.1)
            max_len = int(seq_length * mask_ratio)
            if max_len > min_len:
                block_len = torch.randint(min_len, max_len, (1,)).item()
                start = torch.randint(0, seq_length - block_len, (1,)).item()
                mask[i, start : start + block_len, :] = False

        # --- Random Point Masking ---
        # 30% probability to randomly mask points until we reach the target_masked_count
        if strategy <= 0.3 or strategy > 0.7:

            current_masked_count = total_elements - mask[i].sum().item()
            remaining_to_mask = target_masked_count - current_masked_count

            if remaining_to_mask > 0:

                unmasked_indices = torch.nonzero(mask[i].view(-1)).squeeze()

                if unmasked_indices.numel() > 0:
                    num_to_select = min(remaining_to_mask, unmasked_indices.numel())
                    rand_idx = torch.randperm(unmasked_indices.numel())[:num_to_select]
                    indices_to_mask = unmasked_indices[rand_idx]
                    mask[i].view(-1)[indices_to_mask] = False

    return mask


def inject_compound_noise_torch(V, I, noise_level_base=0.02):
    """
    Inject compound noise into voltage (V) and current (I) tensors to simulate realistic sensor imperfections.
    """
    device = V.device
    v_noisy = V.clone().to(torch.float32)
    i_noisy = I.clone().to(torch.float32)

    # ================= Voltage destructifve =================
    # 1. 绝对高斯白噪声 (对应真实的 ADC 量化底噪和基础热噪声)
    #
    v_noisy += torch.randn_like(V) * noise_level_base

    # 2. 绝对低频随机游走漂移 (对应真实的运放温漂，随时间缓慢累积)
    # 每步累积极其微小的漂移，防止长序列走飞
    drift_step = torch.randn_like(V) * (noise_level_base * 0.05)
    v_noisy += torch.cumsum(drift_step, dim=1)

    # 3. 工业级脉冲尖峰 (对应继电器吸合/高压打火造成的瞬态电磁脉冲)
    # 极低概率 (0.5%)，但幅度固定在极其危险的物理区间 (例如 0.2V ~ 0.5V 之间的绝对突变)
    spike_mask = (torch.rand_like(V) < 0.005).float()
    spike_direction = torch.sign(torch.randn_like(V))
    # 绝对突变幅度：基础噪声的 10 倍到 25 倍
    spike_magnitude = (torch.rand_like(V) * 15 + 10) * noise_level_base
    v_noisy += spike_mask * spike_direction * spike_magnitude

    # ================= 电流 I 的破坏 (基于量程的相对噪声) =================
    # 霍尔电流传感器量程跨度大（如 50A），使用绝对值不合理。
    # 我们提取当前 batch 局部的电流最大值，注入 0.5% 的白噪声。
    B = I.shape[0]
    i_max = torch.max(torch.abs(I).view(B, -1), dim=1)[0].view(B, 1, 1) + 1e-6
    i_noisy += torch.randn_like(I) * (i_max * 0.005)

    return v_noisy, i_noisy


def inject_destructive_real_noise_torch(V, I):
    """
    Inject the formal destructive corruption with fixed protocol parameters.

    Uses the formal destructive protocol:
      voltage white noise sigma = 0.1 V
      voltage random-walk step sigma = 0.005 V
      spike probability = 0.005, magnitude U(1.0, 2.5) V
      current white noise = 0.005 * per-sample max |I|
    """
    v_noisy = V.clone().to(torch.float32)
    i_noisy = I.clone().to(torch.float32)

    v_noisy += torch.randn_like(V) * 0.1
    drift_step = torch.randn_like(V) * 0.005
    v_noisy += torch.cumsum(drift_step, dim=1)

    spike_mask = (torch.rand_like(V) < 0.005).float()
    spike_direction = torch.sign(torch.randn_like(V))
    spike_magnitude = torch.rand_like(V) * 1.5 + 1.0
    v_noisy += spike_mask * spike_direction * spike_magnitude

    B = I.shape[0]
    i_max = torch.max(torch.abs(I).view(B, -1), dim=1)[0].view(B, 1, 1) + 1e-6
    i_noisy += torch.randn_like(I) * (i_max * 0.005)

    return v_noisy, i_noisy


def visual_true_pred_input_with_mask(v_true, v_recon, v_obs, mask, name):
    """
    符合原始逻辑的优化绘图
    v_true, v_recon, v_obs: 已经 flatten 后的 1D numpy array
    mask: 对应的 mask 1D numpy array (置信度)
    """
    t = np.arange(len(v_true))

    # 创建画布：上下两部分，高度比为 3:1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- 上图：信号恢复对比 ---
    # 1. 真值：深灰色虚线，作为物理基准
    ax1.plot(t, v_true, color='#7f8c8d', linestyle='--', label='Ground Truth', alpha=0.8, linewidth=1.5)

    # 2. 观察值：针对“丑”的优化
    # 我们不直接画折线图（因为有随机跳变），而是画散点
    # 正常点用灰色，极低置信度点(mask < 0.1)用浅红色
    is_missing = mask < 0.1
    ax1.scatter(t[~is_missing], v_obs[~is_missing], color='gray', s=8, alpha=0.3, label='Observed')
    ax1.scatter(t[is_missing], v_obs[is_missing], color='#e74c3c', s=8, alpha=0.2, label='Missing/Invalid')

    # 3. 修复值：蓝色实线，加宽突出显示
    ax1.plot(t, v_recon, color='#3498db', label='BatteryDiT Recon', linewidth=2)

    ax1.set_ylabel('Voltage (V)')
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    ax1.set_title('Battery Signal Reconstruction Analysis', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 下图：Mask 置信度展示 ---
    # 使用填充图，直观反映模型对哪些区域“不放心”
    ax2.fill_between(t, 0, mask, color='#f39c12', alpha=0.4, label='Confidence (Mask)')
    ax2.plot(t, mask, color='#d35400', linewidth=1)

    ax2.set_ylim(-0.05, 1.1)
    ax2.set_ylabel('Mask')
    ax2.set_xlabel('Sequence Steps')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.close()
