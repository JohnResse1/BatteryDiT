#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT/source_code"
export PYTHONPATH="$PROJECT/source_code"
echo "NODE=$(hostname)"
echo "RUN_START=$(date -Is)"
python main.py --task_name PopulationIndividual --is_training 0 --model_id seed2021 \
  --recon_model batterydit --predictor trb --data tju_ncm_45c --seed 2021 \
  --stress_seed 3407 --eval_batch_size 128 --num_workers 0 \
  --recn_chkpt "$PWD/checkpoints/Reconstruction/tju_ncm_45c/Reconstruction_seed2021_batterydit_tju_ncm_45c_0/checkpoint.pth" \
  --pred_chkpt "$PWD/checkpoints/PredictDownstream/tju_ncm_45c/PredictDownstream_seed2021_batterydit_tju_ncm_45c_0/best.pth"
echo "RC=$?"
echo "RUN_END=$(date -Is)"
