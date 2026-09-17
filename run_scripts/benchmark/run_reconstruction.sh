#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mit}"
SEED="${2:-2021}"
MODEL="${3:-batterydit}"
EPOCHS=200
BATCH_SIZE=64
SAVE_TRAJECTORIES="${4:-true}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPOSITORY_ROOT/source_code"
if [[ "$MODEL" == "kalman" ]]; then
  python main.py \
    --task_name Reconstruction --is_training 0 \
    --model_id "seed${SEED}" --recon_model "$MODEL" --data "$DATASET" --seed "$SEED" \
    --batch_size "$BATCH_SIZE" --eval_batch_size 128 --num_workers 0 \
    --save_reconstructions "$SAVE_TRAJECTORIES"
  exit 0
fi

python main.py \
  --task_name Reconstruction --is_training 1 \
  --model_id "seed${SEED}" --recon_model "$MODEL" --data "$DATASET" --seed "$SEED" \
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --eval_batch_size 128 --num_workers 0 \
  --validation_interval 10 --validation_per_cell 20 --patience 10 --worst_weight 0.5 \
  --save_reconstructions "$SAVE_TRAJECTORIES"

# usage: ./run_scripts/benchmark/run_reconstruction.sh <dataset> <seed> <batterydit|transformer|csdi|kalman> [save_trajectories]
