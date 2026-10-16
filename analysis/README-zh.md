# 论文分析与绘图（分层）

分析层与绘图层已经解耦，各自只有一个方向的数据流：

```text
results/                 正式实验产物（source_code 训练/评估写出）
   │
   └──> analysis/         计算层：统计、bootstrap、表格
           ├──> analysis/outputs/   绘图层的唯一输入
           └──> ../paper_results/   按图表归档的投稿/审稿数据

analysis/outputs/ ─────> figures/<figure_id>/   绘图层：只画图
```

**硬性约束**（由 `tests/regression/test_analysis_layering.py` 强制）：

- 计算层（`analysis/`）不 import matplotlib；
- 绘图层（顶层 `figures/`）不读 `results/`，不 import scipy/sklearn。

## 目录

| 路径 | 作用 |
| --- | --- |
| `main.py` | 统一入口，使用 `--task` 与 `--list` |
| `utils/` | 路径、表格 IO、结果读取契约 |
| `workflows/` | R3/R4/R5 机制工作流 |
| `analysis/*.py` | 统计与结果归档任务：读取 `results/`，写入 `outputs/` |
| `../figures/` | 主图与补充图代码（Fig. 2–6、S1–S10） |
| `outputs/` | 计算层产物与 canonical workflow 归档 |
| `../paper_results/` | 按图表归档的数据，供投稿与审稿 |

## 用法

从仓库根目录执行：

```bash
# 查看可用任务
python -m analysis.main --list

# 1) 计算层：results/ -> analysis/outputs/
python -m analysis.main --task benchmark
python -m analysis.main --task stress_test
python -m analysis.main --task multi_dataset
python -m analysis.main --task population_individual
python -m analysis.main --task supplementary
python -m analysis.main --task workflow:cross_system       # 机制工作流
python -m analysis.main --task workflow:cross_method
python -m analysis.main --task workflow:longitudinal
python -m analysis.main --task workflow:sliding_occlusion   # 需要 GPU

# 2) 归档：analysis/outputs/ + results/ -> paper_results/
python -m analysis.main --task paper_results

# 3) 绘图层：analysis/outputs/ -> figures/<figure_id>/
python -m figures.main --list
python -m figures.main --figure fig2
python -m figures.main --figure figS6
```

图中的显示标签使用 `Downsampling`、`Random-point missing`、`Segment missing`
和 `Destructive noise`；数据目录和实验协议中的内部 condition key 仍是
`downsample`、`random`、`segment` 和 `destructive`。

## 当前补充图编号

| 文件夹 | 内容 |
| --- | --- |
| `figS1` | reconstruction examples |
| `figS2` | per-cycle waveform-error distributions |
| `figS3` | SOH parity |
| `figS4` | downstream comparison |
| `figS5` | stress reconstruction trajectories |
| `figS6` | battery-level Population/Individual recovery |
| `figS7` | R4 longitudinal, spatial, and prognostic properties |
| `figS8` | external representative trajectory recovery |
| `figS9` | Individual-fidelity/downstream-error association |
| `figS10` | PC-mask routing schematic |

## 数据完整性

计算层在输入不完整时**直接报错并指出缺哪个 setting**，不会静默出图。
例如 `stress_test` 要求四个模型 × 三个 seed 的正式 stress 结果齐全；
缺 CSDI 时会报 `CSDI->TRB requires 3 stress result(s); found model seeds []`。

## 正式 manifold 接受判据

`manifold_gate_passed` 保持原严格定义（要求整电池 bootstrap CI 排除 0）。
另有一个与上游七系统认证一致的接受层级，记录在 `manifold_status`：

- `STRONG`：整体改善且 ≥70% 验证电池改善；
- `VALID`：整体改善且 >50% 验证电池改善；
- `INVALID`：否则（分解停止）。

只有 `INVALID` 会停止分解；使用哪一层级写入 `summary.json`，不会静默替换。

## R3/R4/R5 机制工作流

正式机制结果的执行代码集中在 `analysis/workflows/`，输出写入
`analysis/outputs/canonical/<workflow>/<run_id>/`，按 `raw/`、`analysis/`、
`plotting/` 和 `provenance/` 分层。R4 预后使用冻结的
`dSOH_100 = SOH(cycle+100) - SOH(cycle)`，只保留同一电池存在精确未来
cycle 的样本；模型在 train+validation 电池上拟合，在 test 电池上评估。

正式任务脚本在 `analysis/formal_runs/`，提交入口在
`scripts/submit_formal_runs_node1.sh` 与 `scripts/submit_formal_runs_node2.sh`。
所有主图和补充图代码只放在顶层 `figures/`；`analysis/` 只负责统计和结果归档。
