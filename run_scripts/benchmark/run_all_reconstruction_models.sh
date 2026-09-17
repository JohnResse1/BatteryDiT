#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mit}"
SEED="${2:-2021}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for MODEL in batterydit transformer csdi; do
  "$REPOSITORY_ROOT/run_scripts/benchmark/run_reconstruction.sh" \
    "$DATASET" "$SEED" "$MODEL" true
done

if [[ "$SEED" == "2021" ]]; then
  "$REPOSITORY_ROOT/run_scripts/benchmark/run_reconstruction.sh" "$DATASET" "$SEED" kalman true
fi

# usage: ./run_scripts/benchmark/run_all_reconstruction_models.sh <dataset> <seed>
