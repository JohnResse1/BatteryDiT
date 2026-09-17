# Formal experiments

`main.py` remains the single experiment entry point.  It constructs a task from
`TASK_REGISTRY`, then calls `train(setting)`, `vali(setting)`, and `test(setting)`.

`MultiDataset` and `PopulationIndividual` are registered alongside the core
experiment tasks. They reuse the runners below rather than duplicating
model code.

## Reconstruction (`Reconstruction`)

This is the repository's implementation of the `benchmark_819` reconstruction
protocol.

- Models: `batterydit`, `csdi`, `transformer`, and evaluation-only `kalman`.
- Train data: `train/clean`.
- Reconstruction checkpoint: validation-selected `checkpoint.pth`. Every validation battery contributes 20 equally spaced cycles under all five views; selection minimizes mean waveform RMSE plus `0.5` times the worst-condition RMSE, with patience 10 validation rounds.
- Test output reports both cycle-weighted `sample_rmse_mean` and the paper-primary `battery_balanced_rmse`.
- Test views: `clean`, `downsample`, `segment`, `random`, `destructive`.
- Pairing: every corrupted NPZ is matched to its clean NPZ using its relative
  `<cell>/<cycle>.npz` key; loader ordering is never used as a pairing rule.
- BatteryDiT objective is fixed to pointwise velocity MSE plus `1.0` times the
  physical-voltage first-difference MSE.  There is no no-gradient BatteryDiT
  experiment in the formal runner.
- Formal seed set for full reports: `2021`, `2022`, `2023`.

Checkpoints are written to `source_code/checkpoints/<task>/<dataset>/<setting>/`.
Metrics and representative curves are written to
`results/<task>/<dataset>/<setting>/`. The multi-dataset runner uses its fixed
`results/multi_dataset/` namespace so all dataset-by-seed runs remain together.

Pass `--save_reconstructions true` when complete trajectories are needed for paper figures. It
writes one `reconstructions_<condition>.npz` per test view with aligned
corrupted inputs, clean targets, reconstructions, SOH/RUL labels, and
`<cell>/<cycle>.npz` keys. This is an evaluation artifact, never training
input.

Example:

```bash
cd source_code
python main.py \
  --task_name Reconstruction --is_training 1 --recon_model batterydit --data mit \
  --model_id seed2021 --seed 2021 --epochs 200 --batch_size 64 \
  --learning_rate 1e-4 --ocv_table_path models/ocv/mit.npz
```


## Multi-dataset runs

Multi-dataset evaluation does not have a separate dataset-specific experiment class. Run the
same `Reconstruction` and `PredictDownstream` tasks once per formal dataset.
At startup `main.py` resolves the dataset name, physics profile, voltage range,
and its fixed `models/ocv/<dataset>.npz` table. This prevents an external run
from accidentally using the MIT PO or OCV prior. Cross-dataset tables belong
in `results/`; `analysis/` reads those tables to make paper figures.

## Reconstruction stress test (`ReconstructionStressTest`)

This evaluation-only task loads a formal reconstruction checkpoint and applies
the frozen five-level `10/30/50/70/90` stress protocol to `test/clean`.  It
reports sample-level physical-voltage RMSE for downsample, destructive noise,
random missing, segment missing, and hybrid corruption.

## Downstream prediction (`PredictDownstream`)

This task loads a fixed reconstruction checkpoint through `--recn_chkpt`,
freezes it, and trains the formal TRB predictor.  Set `--predictor tcn` for
the BatteryDiT→TCN comparator. `SOH` and `RUL` are read
directly from formal NPZ files; SOH is already nominal-normalized and must not
be divided again.  RUL is an auxiliary training target.  The selected
checkpoint minimizes the unweighted mean validation SOH RMSE across all five
corruption views, and formal reporting gives
SOH metrics for all five paired test conditions.

## Direct prediction (`DirectPrediction`)

This is the no-reconstruction branch in the paper benchmark. It trains one of
`lstm`, `gru`, `transformer`, `tcn`, or one-channel raw `trb` directly on the
corrupted voltage/current input mixture. It uses the same clean train split,
five validation views, auxiliary RUL target, and validation-SOH checkpoint
selection as the reconstruction-assisted predictor.

## Prediction stress test (`PredictStressTest`)

This eval-only task requires `--recn_chkpt` and `--pred_chkpt`. It runs the
same fixed five-level stress corruption protocol on clean test samples and
reports SOH-only metrics after reconstruction and TRB prediction.

## Degradation signature (`DegradationSignature`)

This eval-only task exposes the existing 128-dimensional TRB representation,
fits a StandardScaler and Ridge(alpha=1.0) probe on balanced clean training
cycles, and reports pooled Kendall tau-b on balanced test cycles. The fixed
clean basis is applied to clean, corrupted, and reconstructed trajectories.

## Population and individual recovery (`PopulationIndividual`)

This eval-only task loads frozen BatteryDiT and TRB checkpoints and evaluates
the full five-topology × L1–L5 stress grid. The population manifold and three
physical degradation signatures are selected only with clean train/validation
batteries. Outputs contain cycle-, battery-, and condition-level `E_pop`,
`E_ind`, `A_ind`, `Q`, `C_ind`, SOH error, and degradation-signature error.
