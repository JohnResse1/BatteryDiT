#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:-/public/home/ymhu/vscode/BatteryDiT_clean_rebuild}"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/source_code"
echo "NODE=$(hostname)"
echo "RUN_START=$(date -Is)"
python -m analysis.workflows.main --workflow r5_threshold_sensitivity
echo "RC=$?"
echo "RUN_END=$(date -Is)"
