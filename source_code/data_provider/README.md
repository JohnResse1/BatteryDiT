# Data provider

`BatteryDataset` is the only reader for formal data. It accepts exactly one of:

```text
mit, hust, mich, sdu, stanford_2,
tju_nca_45c, tju_ncm_45c, tju_ncm_nca_25c
```

Each item is returned as:

```python
I, V, SOH, RUL
```

with shapes `(1000, 1)`, `(1000, 1)`, `(1,)`, `(1,)`. The loader validates those shapes and rejects files that do not contain exactly these fields.

A provider reads one dataset, one split, and one condition. It never mixes chemistry cohorts or label policies. `train` always uses `clean`; `vali` and `test` use the requested condition. MIT accepts the historical `destructive` spelling and resolves it to `destructive`.

The provider does not create corruptions. If a requested condition is absent, preprocessing has not built it and the loader raises an error that lists available conditions.

## I/O characteristics

The formal schema stores one compressed NPZ per cycle. This makes every sample
independently auditable and keeps clean/corrupted pairing explicit, but large
cohorts perform many small-file reads during training. HUST contains 101,550
training cycles and Stanford_2 contains 127,636, compared with 17,583 for MIT.
Their 200-epoch runs are therefore expected to take substantially longer.

`num_workers` changes loading throughput only; it does not change samples,
labels, order keys, or corruption views. On a shared filesystem, benchmark the
worker count before a formal batch and record the final value in `config.json`.
