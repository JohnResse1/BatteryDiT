# Checkpoints

This directory stores model state files only. Core benchmark runs use the same
relative `<task>/<dataset>/<run_id>/` path as their matching directory under
`results/`. The multi-dataset runner groups metrics under
`results/multi_dataset/` while retaining task-specific checkpoint paths.

Example:

```text
checkpoints/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/checkpoint.pth
results/Reconstruction/mit/Reconstruction_seed2021_batterydit_mit_0/test_metrics.json
```


Reconstruction uses `checkpoint.pth`, selected by the fixed five-view
validation score: mean waveform RMSE plus `0.5` times the worst-condition
RMSE. Downstream TRB uses `best.pth`, selected every ten epochs by the mean
validation SOH RMSE across all five corruption views. `last.pth` is a recovery
artifact and is never substituted for the validation-selected formal model.
