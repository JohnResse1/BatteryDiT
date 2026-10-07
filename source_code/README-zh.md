# Source code：实现与实验接口说明

完整复现流程请先阅读仓库根目录的
[`REPRODUCTION.md`](../REPRODUCTION.md)。本文件保留代码目录内的数据格式、
模型协议和实验依赖说明。

本目录是 BatteryDiT 的唯一代码入口。所有正式实验均从[`main.py`](main.py) 启动。

## 目录与产物

```text
source_code/
├── main.py                 # 所有正式模型实验的统一入口
├── data_preprocessing/     # raw sources → 正式 per-cycle NPZ 的离线构建
├── data/                   # raw_sources 与生成后的正式 NPZ（本地大文件）
├── data_provider/          # 唯一的正式数据读取与 clean/corrupt 配对逻辑
├── models/                 # BatteryDiT、PO/OCV、重建基线与 TRB
├── exp/                    # 正式实验 runners
├── configs/                # 固定的多数据集协议
└── checkpoints/            # 本地模型权重
```

实验结果写入仓库根目录的 `results/`：

```text
results/<task>/<dataset>/<setting>/
```
供后续分析直接读取。

`analysis/` 只读取这些结果来生成论文表格和图片，不负责训练或改写模型输出。

所有命令都应在本目录执行：

```bash
cd source_code
python main.py --help
```

`--use_gpu false` 可用于 CPU 调试；正式训练使用可用的 GPU。

## 数据与标签

正式样本的唯一格式为：

```text
data/<dataset>/<split>/<condition>/<cell>/<cycle>.npz
```

每个 NPZ 只含 `I`、`V`、`SOH`、`RUL`：

- `I`、`V` 是 `(1000, 1)` 的 `float32` 曲线。
- `SOH` 已是最终监督标签，训练时不能再次除以 nominal capacity。
- `RUL` 是辅助预测目标，定义为
  `(last_valid_cycle - current_cycle) / (last_valid_cycle - first_valid_cycle)`。
- `train` 只包含 `clean`；`vali` 和 `test` 都包含严格配对的
  `clean`、`downsample`、`segment`、`random`、`destructive` 五个视图。

数据层保留的可构建数据集为：

```text
mit, hust, mich, sdu, stanford_2,
tju_nca_45c, tju_ncm_45c, tju_ncm_nca_25c
```

具体的数据来源、引用及标签政策见 [`data/README.md`](data/README.md)。由于原始数据集较大，请从官方公开来源下载，并按说明放入 `source_code/data/raw_sources/`。

## 从原始数据重建正式 NPZ

原始输入存放在 `data/raw_sources/`，预处理只在离线阶段运行：

```bash
# 重建一个数据集
python -m data_preprocessing.main --dataset mit --replace

# 重建全部可构建数据集
python -m data_preprocessing.main --dataset all --replace
```

构建先写入 `data/.<dataset>.building/`，仅在成功后发布到
`data/<dataset>/`。各 adapter 的共同输出约束、1000 点重采样和固定 split
都在 `data_preprocessing/`；训练过程不会导入该目录。

## PO / SOC--OCV prior

每个数据集的固定 SOC--OCV 表位于：

```text
models/ocv/<dataset>.npz
```

该表只由相应数据集的 `train/clean` 计算一次，训练、验证与测试均只读取
该固定结果。主入口会依据 `models.dataset_physics` 自动选择化学体系、
电压范围与 OCV 表；正常运行不要手工替换为其他数据集的 prior。

## 正式实验

### 正式 benchmark

正式 benchmark 包含 reconstruction、冻结 reconstruction 后的 predictor，以及
不经过 reconstruction 的 direct predictor 三条链路。

第一步训练并测试 reconstruction：

```bash
python main.py \
  --task_name Reconstruction --is_training 1 \
  --model_id seed2021 --recon_model batterydit --data mit --seed 2021 \
  --epochs 200 --batch_size 64 --learning_rate 1e-4
```

重建训练只使用 `train/clean`。BatteryDiT 的正式损失为 velocity MSE 加上
权重为 1.0 的 physical-voltage 一阶差分 MSE。每 10 个 epoch 在五种
validation view 上评估；每个 validation 电池固定抽取 20 个等距 cycle，
并以“平均 waveform RMSE + 0.5 × 最差条件 RMSE”选择 checkpoint，连续
10 次 validation 未改善则提前停止。测试同时保存 sample-level 与论文使用的
battery-balanced waveform RMSE。最佳 epoch 写入：

```text
checkpoints/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/checkpoint.pth
```

第二步冻结该 checkpoint 并训练 TRB：

```bash
python main.py \
  --task_name PredictDownstream --is_training 1 \
  --model_id seed2021 --recon_model batterydit --data mit --seed 2021 \
  --epochs 200 --batch_size 64 --learning_rate 1e-4 --weight_decay 1e-4 \
  --recn_chkpt checkpoints/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/checkpoint.pth
```

TRB 训练时只更新 predictor 参数。每 10 epoch 分别计算五种 `vali` 视图的
SOH RMSE，以五者的非加权平均选择 `best.pth`；最终报告五种 `test` 视图。
传入 `--predictor tcn` 可运行 BatteryDiT→TCN 对照。

第三步是不经过 reconstruction 的 direct baselines：

```bash
python main.py \
  --task_name DirectPrediction --is_training 1 \
  --model_id seed2021 --direct_model lstm --data mit --seed 2021 \
  --epochs 200 --batch_size 64 --learning_rate 1e-4
```

`--direct_model` 可取 `lstm`、`gru`、`transformer`、`tcn` 或 `trb`。前四者
使用原始 `[V,I]` 双通道，raw-TRB 仅使用原始 `V`；五种模型均以五视图 validation
SOH RMSE 选择 checkpoint 后再测试。

### 压力测试

压力测试从 `test/clean` 开始，在内存中以固定随机种子生成五档
10/30/50/70/90 的扰动强度。它不会训练或改写已有 checkpoint。

```bash
# 重建压力测试
python main.py \
  --task_name ReconstructionStressTest --is_training 0 \
  --model_id seed2021 --recon_model batterydit --data mit --seed 2021 \
  --recn_chkpt checkpoints/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/checkpoint.pth

# SOH 压力测试
python main.py \
  --task_name PredictStressTest --is_training 0 \
  --model_id seed2021 --recon_model batterydit --data mit --seed 2021 \
  --recn_chkpt checkpoints/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/checkpoint.pth \
  --pred_chkpt checkpoints/PredictDownstream/mit/PredictDownstream_seed2021_batterydit_trb_mit_0/best.pth
```

### 多数据集评估

多数据集评估的协议固定在
[`configs/formal_protocols/multi_dataset_reconstruction.json`](configs/formal_protocols/multi_dataset_reconstruction.json)。
每个数据集与每个 seed 独立训练，不进行数据集混合训练。

```bash
# 先完成所有 reconstruction
python main.py --task_name MultiDataset --is_training 1 --multi_stage reconstruction

# 使用已有的 matching reconstruction checkpoint 训练全部 TRB
python main.py --task_name MultiDataset --is_training 1 --multi_stage downstream

# 或对选定数据集和 seed 连续执行两步
python main.py --task_name MultiDataset --is_training 1 --multi_stage all \
  --multi_datasets mit hust --multi_seeds 2021
```

论文的 cross-system 协议默认覆盖七个 cohort（MIT、HUST、MICH、SDU、Stanford、
TJU-NCA、TJU-NCM）与 seeds `2021/2022/2023`。`tju_ncm_nca_25c` 可在数据层单独
构建，但不进入论文 Fig.4 的七 cohort。下游 checkpoint 仍按五种 validation view
的平均 SOH RMSE 选择。

### Population versus individual

该计算不训练 BatteryDiT 或 TRB，而是读取对应数据集已经训练好的 frozen checkpoint，
在固定 observation seed 3407 下统一生成五种 corruption × L1–L5，并完成分解：

```bash
bash ../run_scripts/population_individual/run.sh mit 2021
```

runner 使用 `train/clean` 拟合 population manifold、以 `vali/clean` 选择 spline/Ridge
超参数，并与常数基线进行 whole-battery bootstrap gate；gate 未通过时明确
停止 individual 分解。gate 通过后，输出 per-cycle 与 battery-level 的
`E_pop`、`E_ind`、`A_ind`、`Q`、`C_ind`、waveform RMSE、SOH error 和
degradation-signature error。六体系正式批处理见
`slurm_scripts/population_individual/run_six_systems.sbatch`；统计、bootstrap 与绘图
由 `analysis/` 完成。

## 复现检查顺序

建议按下面的依赖顺序运行和检查：

1. 重新生成或核验 `data/<dataset>/` 的 NPZ、固定 split 与五种视图配对。
2. 核验 `models/ocv/<dataset>.npz` 已存在且来自相同数据集的 `train/clean`。
3. 运行 reconstruction，检查最终 `checkpoint.pth`、五视图 metrics。
4. 运行 TRB，检查 `best.pth` 与五视图 validation/test metrics。
5. 在相同 reconstruction checkpoint 上运行压力测试，或使用多数据集批量入口。
6. 在 frozen BatteryDiT/TRB checkpoint 上运行六体系 population/individual stress 分解。

各子目录的详细约束见：

- [`data_preprocessing/README-zh.md`](data_preprocessing/README-zh.md)
- [`data_provider/README.md`](data_provider/README.md)
- [`models/README.md`](models/README.md)
- [`exp/README.md`](exp/README.md)
