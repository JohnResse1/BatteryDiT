# 机制结果工作流（results 3/4/5）

本目录承担论文叙事中"population–individual 分解 + 机制分析"的全部结果。
所有工作流只读取：

- 认证分解 `source_code/exp/population_individual.py`；
- 正式产物 `results/<task>/<dataset>/<setting>/`。

**不重训模型、不重拟合 manifold、不重新定义指标。** 唯一做前向计算的是
`sliding_occlusion_runner`，它用冻结的正式 checkpoint 做遮挡重建。

## 入口

从仓库根目录执行：

```bash
python -m analysis.workflows.main --workflow <name>
```

| workflow | 内容 | 资源 |
| --- | --- | --- |
| `cross_system` | 六系统条件化 population/individual 不对称（R5-1..R5-4） | CPU |
| `cross_method` | BatteryDiT / Transformer / Kalman / CSDI 跨方法不对称（R5-5） | CPU |
| `r5_framework_generality` | 七 cohort reconstruction / waveform / downstream provenance（R5-A） | CPU |
| `r5_threshold_sensitivity` | 六系统 P90/P95/P99 population-success conditional analysis（R5-D/G） | CPU |
| `longitudinal` | 单电池纵向 individual 动力学与个性化预后（R4-1/5/6/7/8） | CPU |
| `sliding_occlusion` | 滑窗遮挡 + support–vulnerability + waveform–individual（R4-3/R4-4） | GPU |
| `r4_prognosis` | `dSOH_100` 的 P/P+I/P+H/P+H+I 预后对照（R4-06～R4-09） | CPU |
| `r4_reconstruction_prognosis` | MIT L5 重建后的 fidelity–prognosis 联动（R4-10/R4-11） | CPU/GPU |
| `ablation_strict_benchmark819` | MIT benchmark-819 的四行 PO/gate component ablation | GPU |

R4 正式预后统一使用同一电池的精确 `cycle+100` 目标；缺失未来 cycle 的行排除，
train+validation 电池拟合、test 电池评估。正式执行脚本位于
`analysis/formal_runs/`，绘图代码统一位于仓库顶层 `figures/`；工作流本身不生成论文图。

## Slurm

```bash
sbatch source_code/slurm_scripts/workflows/mechanism_results.sbatch
```

数组 0–4 依次对应上表；数组 4（GPU）耗时最长。

MIT 消融使用独立的依赖链提交到 node2：

```bash
bash scripts/submit_ablation_strict_benchmark819.sh
```

该链包含 9 个独立变体训练任务、12 个固定 test-view 评估任务和 1 个汇总任务。
Full 行复用新仓库已有 checkpoint；三行变体按同一 200 epoch 协议独立训练。输出写入
`analysis/outputs/canonical/ablation_strict_benchmark819/<run_id>/`，四指标表位于
`plotting/ablation_four_metric_table.csv`。

## 关键口径

- 群体成功判据沿用正式规则，不重定义：
  `population_success = E_pop <= tau_pop`，`tau_pop = P95(clean-validation E_pop)`。
  正式 `tau_pop` 仅 MIT 存档；其余系统在同一冻结规则下于各自 clean validation 上物化。
- 互补判据：`E_ind >= 1`、`Q` 落在 `[0.95, 1.05]` 之外、`C_ind <= 0`。
- 所有不确定区间按 **整电池** bootstrap，从不按 cycle。
- 所有 PCA/scaling 只在 TRAIN 上拟合；所有预测器 battery-wise（train+vali 拟合 → test 评估）。

## 输出

R5 canonical workflow 写入 `analysis/outputs/canonical/<workflow>/<run_id>/`，按
`raw/`、`analysis/`、`plotting/`、`provenance/` 分层；绘图代码只能读取
`plotting/`。历史探索入口已迁移到顶层 `finding/`，不会再写入正式输出目录。
