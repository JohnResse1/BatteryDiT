#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/source_code"
echo "NODE=$(hostname)"
echo "RUN_START=$(date -Is)"
python -m analysis.workflows.main --workflow r4_occlusion --device cuda:0
echo "RC=$?"
echo "RUN_END=$(date -Is)"
