#!/usr/bin/env bash
set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-/public/home/ymhu/vscode/BatteryDiT_clean_rebuild}"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/source_code"
python -m analysis.workflows.r5_formal_canonical "$@"
