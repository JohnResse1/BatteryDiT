# Experiment results

本目录只保存科学结果，不保存模型权重。对应的 `.pth` 文件统一位于
`source_code/checkpoints/`。

普通任务在两个输出根目录中使用相同的相对路径：

```text
source_code/checkpoints/<task>/<dataset>/<setting>/
results/<task>/<dataset>/<setting>/
```

例如：

```text
source_code/checkpoints/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/checkpoint.pth
results/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/test_metrics.json
```

一个完整 run 可包含：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 模型、协议和实际命令参数 |
| `history.json` | 每个 epoch 的训练损失与 validation 记录 |
| `selection.json` | checkpoint 选择规则、最佳 epoch 和最佳分数 |
| `validation_metrics.json` | 五种 validation view 的正式指标 |
| `test_metrics.json` | 五种 test view 的正式指标 |
| `examples_<condition>.npz` | 少量可视化样例 |
| `reconstructions_<condition>.npz` | 可选的全量对齐轨迹、标签与 sample key |

压力测试、多数据集和 population/individual 任务会在自己的 task namespace 下保存
相应的 condition-level 或 battery-level 表。`analysis/` 只读取这些正式输出；缺少
模型、seed 或 condition 时会停止，不会用不完整结果生成正式图片。

完整轨迹文件体积较大，属于本地复现产物。用于归档时，应保留对应的 `config.json`、
指标 JSON 和 checkpoint 路径，使每个图片输入都能追溯到具体 run。
