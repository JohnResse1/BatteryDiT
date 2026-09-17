# Formal protocol configurations

`multi_dataset_reconstruction.json` is the command-independent multi-dataset
configuration used for clean-repository reruns. It adopts the benchmark819
model, loss, seed, epoch, and metric conventions.

The old multi-dataset825 predictor selected checkpoints from five stored
validation corruption views. The rebuilt formal data now contains the same
five views (`clean`, `downsample`, `segment`, `random`, and `destructive`). Every
ten epochs, the predictor is selected by their unweighted mean SOH RMSE.

## Data audit

The current formal data have a cell-disjoint fixed split and five validation
and test views.
HUST uses the frozen 54/8/15 cell split from the old formal protocol, yielding
101,550 train, 15,218 validation, and 29,354 samples in each test view. These
counts match the old formal protocol exactly.

MICH (13,615/1,862/3,583), SDU (5,548/776/1,713), and Stanford_2
(127,636/17,162/35,672) also match its retained split counts. TJU is rebuilt
from its raw CSV files using the unified valid-charge rule. It retains a small
number of additional valid end cycles relative to historical cached NPZs:
TJU-NCA is 10,764/1,498/3,666 and TJU-NCM is 11,782/1,475/4,409
(train/validation/test). New TJU results must therefore be reported as a
rebuild, not as bitwise reproduction of the old cached-data results.
