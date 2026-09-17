#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mit}"
SEED="${2:-2021}"
MODEL="${3:-lstm}"
IS_TRAINING="${4:-1}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPOSITORY_ROOT/source_code"
python main.py \
  --task_name DirectPrediction --is_training "$IS_TRAINING" \
  --model_id "seed${SEED}" --direct_model "$MODEL" --data "$DATASET" --seed "$SEED" \
  --epochs 200 --batch_size 64 --learning_rate 1e-4 --weight_decay 0.0 \
  --num_workers 0

# usage: ./run_scripts/benchmark/run_direct_prediction.sh <dataset> <seed> <lstm|gru|transformer|tcn|trb> [is_training]
