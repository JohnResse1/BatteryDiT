# Formal data inventory

Every formal sample uses:

```text
<dataset>/<split>/<condition>/<cell>/<cycle>.npz
```

and contains only `I`, `V`, `SOH`, and `RUL`. `SOH` is already the final supervised target, so training code must not apply capacity normalization again. All formal datasets use fixed-nominal SOH except `sdu`, which uses its fixed second-life per-cell capacity target.

`train` contains `clean` trajectories only. Both `vali` and `test` contain the
same five strictly paired views: `clean`, `downsample`, `segment`, `random`, and
`destructive`.

The retained formal datasets are:

- `mit`
- `hust`
- `mich`
- `sdu`
- `stanford_2`
- `tju_nca_45c`
- `tju_ncm_45c`
- `tju_ncm_nca_25c`

Legacy local TJU cohorts, transfer cohorts, SDU boundary folds, and separate nominal-label views are intentionally absent.

## Raw preprocessing inputs

`raw_sources/` contains the original inputs used to rebuild the formal data offline. Training and the data loader never read this directory.

```text
raw_sources/
├── mit/                       # MIT raw train/validation pkl
└── datasets/
    ├── hust/                  # HUST raw cell pkl
    ├── batterylife/           # MICH and Stanford_2 archives
    ├── sdu/                   # SDU second-life archive
    └── tju/                   # TJU 原始 cycle CSV 与阻抗文件
```

## Data sources and citations

MIT, HUST, MICH, Stanford_2, SDU, and TJU are public-source datasets. They are retained locally only to make preprocessing and reported experiments reproducible. Any use of these data should cite the original source papers below as well as this repository.

MICH and Stanford_2 are read from the **BatteryLife standardized archive**. The archive is an upstream data package rather than a dataset paper: this repository records the exact upstream revision (`9572e47b1d36ecb31fe58f7d2874a7355dbb6fea`) and cites both BatteryLife [4] and each dataset's original paper. SDU is read from its own public second-life archive, not from BatteryLife.

| Formal dataset | Public source | Required citation |
| --- | --- | --- |
| `mit` | Severson et al. early-cycle-life dataset | [1] |
| `hust` | Mendeley Data, `nsc7hnsg4s/2` | [2] |
| `mich` | BatteryLife standardized MICH archive | [3], [4] |
| `stanford_2` | BatteryLife standardized Stanford_2 archive | [4], [5] |
| `sdu` | Zenodo record 14859405 | [6] |
| TJU cohorts | Zenodo record 6405084 | [7] |

## References

1. K. A. Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*. **Nature Energy**, 4, 383–391, 2019. <https://doi.org/10.1038/s41560-019-0356-8>.
2. *Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning*. **Energy & Environmental Science**, 2022. Original HUST data: <https://data.mendeley.com/datasets/nsc7hnsg4s/2>.
3. *Predicting the impact of formation protocols on battery lifetime immediately after manufacturing*. **Joule**, 2021. Original MICH data: <https://www.batteryarchive.org/list.html>.
4. BatteryLife standardized data archive, upstream revision `9572e47b1d36ecb31fe58f7d2874a7355dbb6fea`. Used here for the standardized MICH and Stanford_2 per-cell archives.
5. *Data-driven analysis of battery formation reveals the role of electrode utilization in extending cycle life*. **Joule**, 2024. Original Stanford data: <https://data.matr.io/8/>.
6. *Deep sorting of reused batteries for enabling long-term consistency grouping with unknown prior conditions*. **Cell Reports Physical Science**, 2025. Original SDU data: <https://zenodo.org/records/14859405>.
7. *Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation*. **Nature Communications**, 2022. Original Tongji/TJU data: <https://zenodo.org/records/6405084>.
