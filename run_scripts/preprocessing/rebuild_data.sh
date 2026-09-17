#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: rebuild_data.sh <dataset|all>}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPOSITORY_ROOT/source_code"
python -m data_preprocessing.main --dataset "$DATASET" --replace
