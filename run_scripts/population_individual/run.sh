#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mit}"
SEED="${2:-2021}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECONSTRUCTION_SETTING="Reconstruction_seed${SEED}_batterydit_${DATASET}_0"
# MultiDataset writes PredictDownstream_seed<seed>_batterydit_<dataset>_0, while the
# MIT benchmark archive uses the ..._batterydit_trb_<dataset>_0 form. Prefer whichever exists.
PREDICTION_SETTING="PredictDownstream_seed${SEED}_batterydit_trb_${DATASET}_0"
if [[ ! -f "$REPOSITORY_ROOT/source_code/checkpoints/PredictDownstream/$DATASET/$PREDICTION_SETTING/best.pth" ]]; then
  PREDICTION_SETTING="PredictDownstream_seed${SEED}_batterydit_${DATASET}_0"
fi
RECONSTRUCTION_CHECKPOINT="$REPOSITORY_ROOT/source_code/checkpoints/Reconstruction/$DATASET/$RECONSTRUCTION_SETTING/checkpoint.pth"
PREDICTION_CHECKPOINT="$REPOSITORY_ROOT/source_code/checkpoints/PredictDownstream/$DATASET/$PREDICTION_SETTING/best.pth"

cd "$REPOSITORY_ROOT/source_code"
python main.py \
  --task_name PopulationIndividual --is_training 0 \
  --model_id "seed${SEED}" --recon_model batterydit --predictor trb \
  --data "$DATASET" --seed "$SEED" --stress_seed 3407 --eval_batch_size 128 --num_workers 0 \
  --recn_chkpt "$RECONSTRUCTION_CHECKPOINT" --pred_chkpt "$PREDICTION_CHECKPOINT"

# usage: run.sh <dataset> <model-seed>
