# Data preprocessing

这里是生成 `source_code/data/` 正式训练与评估数据的离线代码。每个输出样本都是一个 cycle，对应一个 NPZ，且只包含：
```text
I, V, SOH, RUL
```

`I` 与 `V` 始终是 `(1000, 1)`的`numpy array`；`SOH` 与 `RUL` 均为单个标量。预处理完成最终标签计算，训练代码不得再次做额定容量或 min--max 归一化。

## 数据来源

MIT 是本地基准数据，独立放在 `sources/mit.py`。其余数据的 adapter 统一位于 `sources/datasets/`：

```text
data_preprocessing/sources/datasets/
├── hust.py                 # HUST 原始 pkl
├── batterylife.py          # MICH 与 Stanford 的 BatteryLife archive
├── sdu.py                  # SDU second-life CSV archive
└── tju.py                  # TJU 原始 per-cycle CSV
```
具体对应数据集的来源请参考引用目录，相应的引用在对应py文件中也有标记。

## 固定的数据协议

1. 从原始记录中提取有效的充电段及该 cycle 的容量。
2. 每条曲线只重采样一次，统一为 1000 点。
3. 在预处理时写入最终 SOH：MIT/HUST/MICH/Stanford/TJU 使用确定的额定容量；SDU 使用每个 cell 的完整 second-life 容量范围做 min--max，这是为了避免second-life电池存在的较大的容量差异造成的过拟合。
4. RUL 对每个 cell 统一定义为：

   ```text
   (last_valid_cycle - current_cycle) / (last_valid_cycle - first_valid_cycle)
   ```

5. `train` 只写入 `clean`；`vali` 和 `test` 都生成 `clean`、`downsample`、`segment`、`random`、`destructive`。这些扰动都在 1000 点重采样之后生成，MIT 的目录名同样为 `destructive`。

固定的 cell membership 保存在 `splits/<dataset>.json`。它不能在构建或训练时随机划分。HUST 使用冻结的 54/8/15 cell 划分。TJU 仅保留具有有效充电段的原始 cycle，再统一重采样并生成扰动视图。

## 运行

从仓库根目录运行：

```bash
python -m source_code.data_preprocessing.main \
  --dataset mit \
  --output-root /path/to/rebuilt_data

python -m source_code.data_preprocessing.main \
  --dataset tju_nca_45c \
  --dataset-root /path/to/dataset_raw_data \
  --output-root /path/to/rebuilt_data
```

每个 dataset 都先构建到 `.<dataset>.building/` staging 目录；成功后才发布到 output root。目标目录已经存在时，命令会拒绝覆盖，只有显式加入 `--replace` 才会在成功构建后替换它。确认 cell split、样本数、clean 曲线、SOH、RUL 和测试条件覆盖均正确后，才可对 `source_code/data/` 使用 `--replace`。

## 数据来源与引用说明

本仓库使用的原始输入数据均来源于公开发布的电池数据集。其中，MICH 和 Stanford_2 使用 BatteryLife 发布的标准化数据归档；MIT、HUST、SDU 和 TJU 则基于各自原始数据源进行处理。

各数据集的具体说明、原始数据链接、BatteryLife 版本信息以及相应的引用要求，统一维护于 source_code/data/README.md以及对应数据集的预处理`<dataset>.py`文件的注释中。
