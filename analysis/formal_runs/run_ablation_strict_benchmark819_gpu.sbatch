#!/usr/bin/env bash
#SBATCH --job-name=abl819_mit
#SBATCH --partition=gpu_48
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/public/home/ymhu/vscode/BatteryDiT_clean_rebuild/analysis/formal_runs/abl819_%A_%a.out
#SBATCH --error=/public/home/ymhu/vscode/BatteryDiT_clean_rebuild/analysis/formal_runs/abl819_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export LC_ALL=C
export LANG=C
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PROJECT=/public/home/ymhu/vscode/BatteryDiT_clean_rebuild
PYTHON=/public/home/ymhu/anaconda3/bin/python
RUNNER=${PROJECT}/analysis/workflows/ablation_strict_benchmark819.py

cd "${PROJECT}"
mkdir -p "${PROJECT}/analysis/formal_runs"
: "${RUN_ROOT:?RUN_ROOT is required; use scripts/submit_ablation_strict_benchmark819.sh}"

case "${RUN_MODE:?RUN_MODE is required}" in
  initialize)
    "${PYTHON}" -u "${RUNNER}" --run-root "${RUN_ROOT}" --device cuda initialize
    ;;
  train)
    "${PYTHON}" -u "${RUNNER}" --run-root "${RUN_ROOT}" --device cuda train \
      --task-index "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  evaluate)
    "${PYTHON}" -u "${RUNNER}" --run-root "${RUN_ROOT}" --device cuda evaluate \
      --task-index "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  aggregate)
    "${PYTHON}" -u "${RUNNER}" --run-root "${RUN_ROOT}" --device cuda aggregate
    ;;
  smoke)
    "${PYTHON}" -u "${RUNNER}" --run-root "${RUN_ROOT}" --device cuda smoke
    ;;
  *)
    echo "Unknown RUN_MODE=${RUN_MODE}" >&2
    exit 2
    ;;
esac
