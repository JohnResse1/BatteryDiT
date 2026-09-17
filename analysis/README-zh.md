# 论文分析与绘图

`analysis/main.py` 是论文图片的统一入口。分析代码只读取仓库根目录下正式实验生成的
`results/`，不会训练模型，也不会修改 `source_code/data/`、checkpoint 或旧仓库。

目录结构：

```text
analysis/
├── main.py                 # 统一入口
├── common/                 # 路径、正式结果读取、绘图与导出规范
├── figures/                # 按科学内容命名的主图代码
└── outputs/
    ├── main/               # 主图 SVG、PDF、600 dpi PNG
    ├── supplementary/      # 补充图
    └── audit/              # 图片尺寸与来源审计
```

正式 benchmark 全部完成后运行：

```bash
cd analysis
python main.py --figure benchmark --dataset mit
```

该图直接读取以下三组结果：

- `results/Reconstruction/mit/`
- `results/PredictDownstream/mit/`
- `results/DirectPrediction/mit/`

脚本要求 BatteryDiT、CSDI 和 Transformer 各有三个 seeds，Kalman reconstruction
有一个确定性结果，其余预测模型均有三个 seeds。任何结果缺失、condition 缺失或指标
出现 NaN 时都会停止并指出具体目录，不会用已有 seeds 静默作图。

压力测试和多数据集图使用同一入口：

```bash
python main.py --figure stress_test --dataset mit
python main.py --figure multi_dataset
python main.py --figure population_individual
```

四类主图都只读取各自正式输出，并强制检查模型、seed、condition 和有限数值。
其中 `stress_test` 还要求三个 `DegradationSignature` readout 完成；
`population_individual` 要求六个 nominal-SOH 系统的正式 25 条件输出完成。

当前已注册的补充结果图同样使用 `--figure`：

```bash
python main.py --figure reconstruction_examples --dataset mit
python main.py --figure waveform_distributions --dataset mit
python main.py --figure soh_parity --dataset mit
python main.py --figure downstream_comparison --dataset mit
python main.py --figure manifold_validation
python main.py --figure individual_fidelity
python main.py --figure signature_linkage
```

它们分别对应论文 Supplementary Fig. 1–4、7–9 的科学内容。文件名按内容命名，
不沿用旧目录中曾经错位的 `FigS*.py` 编号。
