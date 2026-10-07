# Reproduction guide / 复现说明

This is the canonical entry point for reproducing the formal BatteryDiT
experiments. The commands below follow the protocol implemented in this
repository rather than an exploratory notebook workflow.

本文件是 BatteryDiT 正式实验的唯一复现入口。下面的命令以仓库中的正式协议和
脚本为准，不依赖探索性 notebook 或论文图件。

## 1. Scope / 复现范围

The public source release contains the model code, preprocessing code, fixed
protocols, train-only SOC--OCV priors, runners, and regression checks. Raw
datasets, generated NPZ files, checkpoints, result arrays, manuscript files,
and figure files are intentionally kept out of Git.

公开代码仓库包含模型、预处理、固定实验协议、仅由训练集生成的 SOC--OCV prior、
运行脚本和回归检查。原始数据、生成的 NPZ、checkpoint、结果数组、论文文档和图片
不放入 Git，需要在本地准备。

The recommended first rerun is one dataset and one seed:

```text
dataset = mit
seed    = 2021
model   = batterydit
```

正式完整报告使用 seeds `2021`、`2022`、`2023`。每个 `vali` 和 `test` split
都包含严格配对的 `clean`、`downsample`、`segment`、`random` 和
`destructive` 五种视图。

## 2. Environment / 环境

```bash
git clone https://github.com/JohnResse1/BatteryDiT.git
cd BatteryDiT

conda env create -f environment.yml
conda activate batterydit-robust-soh
```

The recorded environment is Python 3.10.9, PyTorch 2.0.1, and CUDA 11.8.
CPU smoke runs are possible by adding `--use_gpu false` to a `source_code/main.py`
command, but formal training is intended for a CUDA device.

## 3. Prepare the raw data / 准备原始数据

Download the public source data listed in
[`source_code/data/README.md`](source_code/data/README.md), and place the raw
files under:

```text
source_code/data/raw_sources/
```

The raw inputs remain local. The formal preprocessing output must have this
layout:

```text
source_code/data/<dataset>/<split>/<condition>/<cell>/<cycle>.npz
```

Each NPZ contains exactly `I`, `V`, `SOH`, and `RUL`. `I` and `V` have shape
`(1000, 1)`. `train` contains only `clean`; `vali` and `test` contain all five
paired views. Dataset-specific input layouts and citations are documented in
[`source_code/data/README.md`](source_code/data/README.md).

Build one dataset from the repository root:

```bash
bash run_scripts/preprocessing/rebuild_data.sh mit
```

Or build directly from `source_code/`:

```bash
cd source_code
python -m data_preprocessing.main --dataset mit --replace
cd ..
```

Use `--dataset all` only after the individual raw-data layouts have been
checked. The builder stages data in a temporary directory and publishes it
only after a successful build; `--replace` replaces an existing local dataset.

The fixed OCV priors under
[`source_code/models/ocv/`](source_code/models/ocv/) are part of the source
release. They are derived from the corresponding dataset's `train/clean` data
and must not be replaced with a prior from another dataset.

## 4. Optional protocol checks / 可选协议检查

From the repository root, run the regression suite with both the source tree
and repository root on `PYTHONPATH`:

```bash
cd source_code
PYTHONPATH=.:.. pytest -q ../tests/regression
cd ..
```

The public checkout contains the versioned regression tests; these checks are
optional and do not require the raw datasets or model checkpoints.

## 5. Train BatteryDiT reconstruction / 训练 BatteryDiT 重建器

Run the formal reconstruction benchmark from the repository root:

```bash
bash run_scripts/benchmark/run_reconstruction.sh mit 2021 batterydit true
```

The final argument requests full reconstructed trajectories. The formal
defaults are 200 epochs, batch size 64, validation every 10 epochs, 20 evenly
spaced cycles per validation battery, and patience 10. The reconstruction
checkpoint is selected using the five-view validation score

```text
mean waveform RMSE + 0.5 × worst-condition RMSE
```

Expected outputs include:

```text
source_code/checkpoints/Reconstruction/mit/
results/Reconstruction/mit/
```

For direct control, the equivalent entry point is
`source_code/main.py --task_name Reconstruction`. See
[`source_code/exp/README.md`](source_code/exp/README.md) for the checkpoint
selection contract.

## 6. Train the downstream TRB / 训练下游 TRB

After the matching BatteryDiT checkpoint exists, train the frozen-reconstruction
TRB predictor:

```bash
bash run_scripts/benchmark/run_downstream.sh mit 2021 batterydit trb
```

This stage loads the selected reconstruction checkpoint, freezes BatteryDiT,
and trains only the predictor. The selected TRB checkpoint is written under:

```text
source_code/checkpoints/PredictDownstream/mit/
results/PredictDownstream/mit/
```

Use `tcn` as the fourth argument for the BatteryDiT-to-TCN comparison. The
direct-prediction baselines do not use a reconstruction checkpoint:

```bash
bash run_scripts/benchmark/run_direct_prediction.sh mit 2021 lstm true
```

## 7. Evaluation and larger runs / 评估与批量运行

Stress tests are evaluation-only and reuse the matching frozen checkpoints:

```bash
bash run_scripts/stress/run_reconstruction_stress.sh mit 2021 batterydit
bash run_scripts/stress/run_prediction_stress.sh mit 2021 batterydit
```

The stress seed is `3407`; the five levels are 10/30/50/70/90. For the
cross-system protocol, run reconstruction first and then the downstream stage:

```bash
bash run_scripts/multi_dataset/run_reconstruction.sh \
  --multi_datasets mit hust --multi_seeds 2021 2022 2023

bash run_scripts/multi_dataset/run_downstream.sh \
  --multi_datasets mit hust --multi_seeds 2021 2022 2023
```

The fixed multi-dataset protocol is recorded in
[`source_code/configs/formal_protocols/multi_dataset_reconstruction.json`](source_code/configs/formal_protocols/multi_dataset_reconstruction.json).
All available runners and their positional arguments are listed in
[`run_scripts/README-zh.md`](run_scripts/README-zh.md).

## 8. BatteryDiT I/O contract / BatteryDiT 输入输出约定

The public reconstruction interface receives physical-unit voltage and current
curves:

```text
V, I: (B, 1000, 1)
```

The internal DiT input is constructed as:

```text
x_base = mask · x_sensor + (1 - mask) · x_po
x_t    = x_base                         # sampling start
DiT input = [x_t, x_po]                 # (B, 1000, 2)
condition = timestep t                  # (B,)
```

During training, `x_t` is the flow interpolation between `x_base` and the
complete target voltage. The mask is used to build `x_base`; it is not passed
as a separate DiT channel. The DiT predicts a voltage flow velocity of shape
`(B, 1000, 1)`. Sampling returns the normalized reconstructed voltage, which
the experiment runner converts back to physical voltage before scoring.

## 9. Reproduction checklist / 复现检查清单

Before treating a run as formal, verify:

1. The dataset was built from the documented public source and has the fixed
   cell split.
2. The matching `models/ocv/<dataset>.npz` prior is present and comes from
   that dataset's `train/clean` data.
3. Reconstruction and downstream prediction use the same dataset and seed.
4. The downstream predictor receives the selected reconstruction checkpoint,
   not an arbitrary epoch.
5. The `config.json`, validation metrics, test metrics, and checkpoint are
   retained together under the corresponding local output directories.

For implementation details, see
[`source_code/README-zh.md`](source_code/README-zh.md) and the component
README files linked from it.
