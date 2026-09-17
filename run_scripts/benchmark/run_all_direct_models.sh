#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mit}"
SEED="${2:-2021}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for MODEL in lstm gru transformer tcn trb; do
  "$REPOSITORY_ROOT/run_scripts/benchmark/run_direct_prediction.sh" "$DATASET" "$SEED" "$MODEL"
done
