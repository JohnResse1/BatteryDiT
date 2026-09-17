# Data preprocessing

This directory contains the offline preprocessing code used to generate the formal training and evaluation data under `source_code/data/`. Each output sample corresponds to one cycle and is stored as a single NPZ file containing only:

```text
I, V, SOH, RUL
```

`I` and `V` are always NumPy arrays with shape `(1000, 1)`, while `SOH` and `RUL` are scalar values. All final label computation is completed during preprocessing. The training code must not perform any additional nominal-capacity normalization or min--max normalization.

## Data sources

MIT is the local benchmark dataset and is handled separately in `sources/mit.py`. Adapters for all other datasets are located under `sources/datasets/`:

```text
data_preprocessing/sources/datasets/
├── hust.py                 # Original HUST pickle files
├── batterylife.py          # BatteryLife archives for MICH and Stanford_2
├── sdu.py                  # SDU second-life CSV archive
└── tju.py                  # Original TJU per-cycle CSV files
```

For detailed information on the source of each dataset, please refer to the citation documentation. The corresponding references are also documented in the header comments of each dataset-specific Python file.

## Fixed data protocol

1. Extract the valid charging segment and the capacity of each cycle from the original records.

2. Each curve is resampled exactly once to a fixed length of 1000 points.

3. The final SOH label is computed during preprocessing. MIT, HUST, MICH, Stanford_2, and TJU use their predefined nominal capacities. For SDU, SOH is min--max normalized using the complete second-life capacity range of each individual cell. This cell-wise definition is used to account for the substantial initial-capacity variation among second-life batteries and to reduce overfitting to inter-cell capacity differences.

4. RUL is consistently defined for each cell as:

   ```text
   (last_valid_cycle - current_cycle) / (last_valid_cycle - first_valid_cycle)
   ```

5. The `train` split contains only `clean` samples. Both `vali` and `test` contain `clean`, `downsample`, `segment`, `random`, and `destructive` conditions. All corruption views are generated after the curves have been resampled to 1000 points. The MIT dataset also uses `destructive` as the corresponding directory name.

Fixed cell memberships are stored in `splits/<dataset>.json`. Cell splits must never be randomly regenerated during either dataset construction or model training. HUST uses the frozen 54/8/15-cell train/validation/test split; its charging segment is the union of the explicit `Constant current charge` and `Constant current-constant voltage charge` states, while every `Constant current discharge_*` state is excluded. For TJU, only original cycles containing a valid charging segment are retained before uniform resampling and corruption-view generation.

## Usage

Run the preprocessing pipeline from the repository root:

```bash
python -m source_code.data_preprocessing.main \
  --dataset mit \
  --output-root /path/to/rebuilt_data

python -m source_code.data_preprocessing.main \
  --dataset tju_nca_45c \
  --dataset-root /path/to/dataset_raw_data \
  --output-root /path/to/rebuilt_data
```

Each dataset is first constructed in a `.<dataset>.building/` staging directory and is published to the output root only after the build completes successfully.

If the target directory already exists, the command refuses to overwrite it by default. Replacement is allowed only when `--replace` is explicitly specified, and the existing dataset is replaced only after the new build has completed successfully.

Before using `--replace` on `source_code/data/`, verify that the cell splits, sample counts, clean curves, SOH labels, RUL labels, and evaluation-condition coverage are all correct.

## Data attribution

All raw inputs used in this repository originate from publicly available battery datasets. MICH and Stanford_2 use the standardized archives distributed by BatteryLife, whereas MIT, HUST, SDU, and TJU are processed from their respective original data sources.

Dataset-specific descriptions, original data links, BatteryLife version information, and the corresponding citation requirements are maintained in `source_code/data/README.md` and in the header comments of each dataset-specific preprocessing `<dataset>.py` file.
