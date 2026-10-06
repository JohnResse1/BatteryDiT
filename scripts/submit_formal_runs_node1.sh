#!/usr/bin/env bash
set -euo pipefail
# Submit Node1 formal analysis runs only.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARTITION="gpu"
NODE="node1"
export PYTHONPATH="$PROJECT/source_code"

submit() {
  local job_name="$1"; shift
  local time_limit="$1"; shift
  local mem="$1"; shift
  local cpus="$1"; shift
  local script="$1"; shift
  sbatch -w "$NODE" -p "$PARTITION" --time="$time_limit" --mem="$mem" \
    --cpus-per-task="$cpus" --job-name="$job_name" \
    --output="$PROJECT/logs/formal_runs/${job_name}_%j.out" \
    --error="$PROJECT/logs/formal_runs/${job_name}_%j.err" "$script"
}
case "${1:-all}" in
  pop_analysis)
    submit formal_pop_analysis 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_pop_analysis.sh";;
  pop_provenance)
    submit formal_pop_provenance 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_pop_provenance.sh";;
  r3_sensitivity)
    submit formal_r3_sensitivity 12:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r3_sensitivity.sh";;
  r4_longitudinal)
    submit formal_r4_longitudinal 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_r4_longitudinal.sh";;
  r4_prognosis)
    submit formal_r4_prognosis 12:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r4_prognosis.sh";;
  r4_reconstruction_prognosis)
    submit formal_r4_reconstruction_prognosis 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r4_reconstruction_prognosis.sh";;
  cross_system)
    submit formal_cross_system 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_cross_system.sh";;
  r5_threshold)
    submit formal_r5_threshold 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r5_threshold.sh";;
  r5_framework_generality)
    submit formal_r5_framework_generality 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_r5_framework_generality.sh";;
  r5_downstream)
    submit formal_r5_downstream 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_cross_system.sh";;
  r5_cross_method)
    submit formal_r5_cross_method 12:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r5_cross_method.sh";;
  all)
    submit formal_pop_analysis 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_pop_analysis.sh"
    submit formal_pop_provenance 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_pop_provenance.sh"
    submit formal_r3_sensitivity 12:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r3_sensitivity.sh"
    submit formal_r4_longitudinal 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_r4_longitudinal.sh"
    submit formal_r4_prognosis 12:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r4_prognosis.sh"
    submit formal_r4_reconstruction_prognosis 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r4_reconstruction_prognosis.sh"
    submit formal_cross_system 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_cross_system.sh"
    submit formal_r5_threshold 08:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r5_threshold.sh"
    submit formal_r5_framework_generality 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_r5_framework_generality.sh"
    submit formal_r5_downstream 04:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_cross_system.sh"
    submit formal_r5_cross_method 12:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r5_cross_method.sh"
    ;;
  *) echo "unknown target $1" >&2; exit 1;;
esac
