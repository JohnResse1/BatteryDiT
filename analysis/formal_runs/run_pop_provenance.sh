#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/source_code"
echo "NODE=$(hostname)"
echo "RUN_START=$(date -Is)"
for dataset in mit hust mich stanford_2 tju_nca_45c tju_ncm_45c; do
  python -m analysis.workflows.main --workflow population_provenance --dataset "$dataset"
done
echo "RC=$?"
echo "RUN_END=$(date -Is)"
