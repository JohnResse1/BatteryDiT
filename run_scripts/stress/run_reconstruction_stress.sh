#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mit}"
SEED="${2:-2021}"
MODEL="${3:-batterydit}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTING="Reconstruction_seed${SEED}_${MODEL}_${DATASET}_0"
RECONSTRUCTION_CHECKPOINT="$REPOSITORY_ROOT/source_code/checkpoints/Reconstruction/$DATASET/$SETTING/checkpoint.pth"

cd "$REPOSITORY_ROOT/source_code"
if [[ "$MODEL" == "kalman" ]]; then
  python main.py \
    --task_name ReconstructionStressTest --is_training 0 \
    --model_id "seed${SEED}" --recon_model "$MODEL" --data "$DATASET" --seed "$SEED" \
    --eval_batch_size 128 --num_workers 0 --stress_seed 3407
else
  python main.py \
    --task_name ReconstructionStressTest --is_training 0 \
    --model_id "seed${SEED}" --recon_model "$MODEL" --data "$DATASET" --seed "$SEED" \
    --recn_chkpt "$RECONSTRUCTION_CHECKPOINT" --eval_batch_size 128 --num_workers 0 --stress_seed 3407
fi

# usage: run_reconstruction_stress.sh <dataset> <seed> <batterydit|transformer|csdi|kalman>
