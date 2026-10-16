#!/usr/bin/env bash
set -euo pipefail

# Submit the two missing analyses to node1.  The source scripts remain in the
# repository under analysis/formal_runs and all outputs are written below
# analysis/outputs/canonical/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT/analysis/formal_runs/logs"
mkdir -p "$LOG_DIR"

SENSITIVITY_JOB=$(sbatch --parsable \
  --nodelist=node1 --partition=gpu --nodes=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=128G --time=3-00:00:00 \
  --job-name=r3_stress_sens \
  --output="$LOG_DIR/r3_stress_sens_%j.out" \
  --error="$LOG_DIR/r3_stress_sens_%j.err" \
  "$PROJECT/analysis/formal_runs/run_r3_stress_sensitivity.sh")

EFFICIENCY_JOB=$(sbatch --parsable \
  --nodelist=node1 --partition=gpu --nodes=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --time=08:00:00 \
  --job-name=comput_efficiency \
  --output="$LOG_DIR/comput_efficiency_%j.out" \
  --error="$LOG_DIR/comput_efficiency_%j.err" \
  "$PROJECT/analysis/formal_runs/run_computational_efficiency_node1.sh")

echo "r3_stress_sensitivity_job=$SENSITIVITY_JOB"
echo "computational_efficiency_job=$EFFICIENCY_JOB"
