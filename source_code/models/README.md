# Models

本目录保存正式实验使用的模型实现。模型层只定义网络、物理算子和 checkpoint
兼容规则；训练循环、数据划分和指标计算均位于 `source_code/exp/`。

## 目录结构

| 目录或文件 | 作用 |
| --- | --- |
| `BatteryDiT.py` | 正式 BatteryDiT 重建模型及统一的 train/sampling 接口 |
| `BatteryDiT_ablation_aligned.py` | 与正式模型接口对齐的保留消融实现 |
| `recon_model/dit_backbone.py` | Diffusion Transformer backbone；参数名属于已有 checkpoint 合约 |
| `recon_model/Baseline_CSDI_official.py` | CSDI 重建基线 |
| `recon_model/Baseline_Transformer.py` | 不使用 metadata 的 Transformer 重建基线 |
| `recon_model/Baseline_Kalman_filter.py` | 无训练参数的 Kalman 重建基线 |
| `predictors/trb/` | 正式 TRB predictor，包含 blocks 与模型入口 |
| `predictors/direct.py` | LSTM、GRU、Transformer、TCN 与 raw-TRB 直接预测基线 |
| `predictors/tcn_reconstruction.py` | BatteryDiT→TCN 对照预测器 |
| `physics_operator.py` | 统一的 physics operator 计算 |
| `po_factory.py` | 根据化学体系构建 physics operator |
| `dataset_physics.py` | 数据集到 PO 类型、电压范围和 OCV prior 的固定映射 |
| `ocv/` | 由各数据集 `train/clean` 一次性提取的 SOC–OCV 表 |
| `checkpoint_loading.py` | 兼容已有重建权重，同时忽略历史 checkpoint 中废弃的下游参数 |

## BatteryDiT 的输入与输出

正式输入是形状为 `(B, 1000, 1)` 的电压 `V` 与电流 `I`。训练模式返回预测
velocity、目标 velocity 和 task id；采样模式返回归一化重建，再由模型内部的
physics operator 转换为物理电压。

重建训练只更新 `model.recon`。TRB 不属于 BatteryDiT checkpoint 的训练目标；下游
实验先加载并冻结重建器，再独立初始化和训练 predictor。因此，历史 checkpoint 中
曾经附带的 predictor 参数不会参与当前模型构建。

## Physics operator 与 OCV

`dataset_physics.py` 是数据集物理配置的唯一注册处。`main.py` 根据数据集自动写入：

- 对应的 physics-operator 类型；
- 合法电压范围；
- `models/ocv/<dataset>.npz`。

每个 OCV 文件只包含 `soc` 与 `voltage`，并且只由同一数据集的 `train/clean`
计算。OCV 提取是离线准备步骤，不在模型训练时重新执行。详细说明见
[`ocv/README.md`](ocv/README.md)。

## Checkpoint 合约

已有 BatteryDiT `.pth` 可以继续加载重建所需参数。加载器要求当前 reconstruction
参数严格匹配，同时允许丢弃历史集成模型中的下游 predictor 参数。新训练保存的
checkpoint 只表达当前正式模型需要的状态。

相关回归检查位于：

```text
tests/regression/test_physics_operator_baseline.py
tests/regression/test_reconstruction_checkpoint_loading.py
tests/regression/test_train_only_ocv_prior.py
```
