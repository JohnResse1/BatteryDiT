"""The fixed reconstruction protocol used for formal benchmark runs."""
from __future__ import annotations

FORMAL_RECONSTRUCTION_CONDITIONS = (
    "clean",
    "downsample",
    "segment",
    "random",
    "destructive",
)
FORMAL_MODEL_SEEDS = (2021, 2022, 2023)
BATTERYDIT_GRADIENT_LOSS_WEIGHT = 1.0

PROTOCOL_NAME = "benchmark_819_reconstruction_v1"

# Training choices belong to the formal protocol rather than shell scripts.
# This keeps every benchmark model reproducible from the same main entry.
RECONSTRUCTION_TRAINING = {
    "batterydit": {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "scheduler_milestones": (),
    },
    "transformer": {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "scheduler_milestones": (),
    },
    "csdi": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "scheduler_milestones": (0.75, 0.90),
    },
}
