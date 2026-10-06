#!/usr/bin/env bash
set -euo pipefail

# Derived-only R4 formal analysis.  This entry point never invokes
# BatteryDiT training/inference or corruption generation.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/source_code:$PROJECT"
python -u -m analysis.workflows.r4_formal_canonical "$@"
