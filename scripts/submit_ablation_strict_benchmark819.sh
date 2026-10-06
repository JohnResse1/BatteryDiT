#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/ymhu/vscode/BatteryDiT_clean_rebuild
RUN_ID=${ABLATION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${PROJECT}/analysis/outputs/canonical/ablation_strict_benchmark819/${RUN_ID}
SBATCH_SCRIPT=${PROJECT}/analysis/formal_runs/run_ablation_strict_benchmark819_gpu.sbatch

mkdir -p "${RUN_ROOT}"

submit_job() {
  local mode=$1
  local dependency=${2:-}
  if [[ -n "${dependency}" ]]; then
    sbatch --parsable \
      --partition=gpu_48 \
      --nodelist=node2 \
      --dependency="afterok:${dependency}" \
      --export="ALL,RUN_ROOT=${RUN_ROOT},RUN_MODE=${mode}" \
      "${SBATCH_SCRIPT}"
  else
    sbatch --parsable \
      --partition=gpu_48 \
      --nodelist=node2 \
      --export="ALL,RUN_ROOT=${RUN_ROOT},RUN_MODE=${mode}" \
      "${SBATCH_SCRIPT}"
  fi
}

initialize_job=$(submit_job initialize)
train_job=$(sbatch --parsable \
  --partition=gpu_48 \
  --nodelist=node2 \
  --array=0-8%3 \
  --dependency="afterok:${initialize_job}" \
  --export="ALL,RUN_ROOT=${RUN_ROOT},RUN_MODE=train" \
  "${SBATCH_SCRIPT}")
evaluate_job=$(sbatch --parsable \
  --partition=gpu_48 \
  --nodelist=node2 \
  --array=0-11%3 \
  --dependency="afterok:${train_job}" \
  --export="ALL,RUN_ROOT=${RUN_ROOT},RUN_MODE=evaluate" \
  "${SBATCH_SCRIPT}")
aggregate_job=$(sbatch --parsable \
  --partition=gpu_48 \
  --nodelist=node2 \
  --dependency="afterok:${evaluate_job}" \
  --export="ALL,RUN_ROOT=${RUN_ROOT},RUN_MODE=aggregate" \
  "${SBATCH_SCRIPT}")

echo "RUN_ROOT=${RUN_ROOT}"
echo "INITIALIZE_JOB=${initialize_job}"
echo "TRAIN_ARRAY=${train_job}"
echo "EVALUATE_ARRAY=${evaluate_job}"
echo "AGGREGATE_JOB=${aggregate_job}"
