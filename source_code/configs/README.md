# Formal experiment configurations

This directory stores command-independent protocol definitions. Configuration
files record choices that must remain identical across datasets and seeds, such
as the model family, optimizer, epoch count, validation views, checkpoint rule,
and reported metric.

Runtime values such as the selected dataset, model seed, resolved OCV path, and
output directory are written to each run's `results/.../config.json`. Formal
data remain under `source_code/data/`, while large checkpoints and trajectory
arrays remain local artifacts.
