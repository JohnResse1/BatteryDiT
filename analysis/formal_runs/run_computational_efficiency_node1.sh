#!/usr/bin/env bash
set -euo pipefail

PROJECT="${BATTERYDIT_PROJECT_ROOT:-/public/home/ymhu/vscode/BatteryDiT_clean_rebuild}"
PYTHON="${PYTHON:-/public/home/ymhu/anaconda3/bin/python}"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/source_code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export LC_ALL=C
export LANG=C

echo "NODE=$(hostname)"
echo "PROJECT=$PROJECT"
echo "PWD=$(pwd)"
echo "RUN_START=$(date -Is)"
"$PYTHON" -u -m analysis.workflows.main \
  --workflow computational_efficiency \
  --device cuda
echo "RUN_END=$(date -Is)"
