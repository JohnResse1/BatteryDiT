#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mit}"
SEED="${2:-2021}"
MODEL="${3:-batterydit}"
PREDICTOR="${4:-trb}"
EPOCHS=200
BATCH_SIZE=64
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTING="Reconstruction_seed${SEED}_${MODEL}_${DATASET}_0"
RECONSTRUCTION_CHECKPOINT="$REPOSITORY_ROOT/source_code/checkpoints/Reconstruction/$DATASET/$SETTING/checkpoint.pth"

cd "$REPOSITORY_ROOT/source_code"
if [[ "$MODEL" == "kalman" ]]; then
  python main.py \
    --task_name PredictDownstream --is_training 1 \
    --model_id "seed${SEED}" --recon_model "$MODEL" --data "$DATASET" --seed "$SEED" \
    --predictor "$PREDICTOR" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --weight_decay "$WEIGHT_DECAY" \
    --eval_batch_size 128 --num_workers 0
else
  if [[ ! -f "$RECONSTRUCTION_CHECKPOINT" ]]; then
    echo "missing reconstruction checkpoint: $RECONSTRUCTION_CHECKPOINT" >&2
    exit 1
  fi
  python main.py \
    --task_name PredictDownstream --is_training 1 \
    --model_id "seed${SEED}" --recon_model "$MODEL" --data "$DATASET" --seed "$SEED" \
    --predictor "$PREDICTOR" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --weight_decay "$WEIGHT_DECAY" \
    --eval_batch_size 128 --num_workers 0 \
    --recn_chkpt "$RECONSTRUCTION_CHECKPOINT"
fi

# usage: ./run_scripts/benchmark/run_downstream.sh <dataset> <seed> <batterydit|transformer|csdi|kalman> [trb|tcn]
