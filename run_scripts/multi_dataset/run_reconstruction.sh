#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT/source_code"
python main.py --task_name MultiDataset --is_training 1 --multi_stage reconstruction \
  --save_reconstructions true "$@"
