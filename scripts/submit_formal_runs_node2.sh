#!/usr/bin/env bash
set -euo pipefail
# Submit Node2 formal GPU/Pop-Individual runs only.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARTITION="gpu_48"
NODE="node2"
submit() {
  local job_name="$1"; shift
  local time_limit="$1"; shift
  local mem="$1"; shift
  local cpus="$1"; shift
  local script="$1"; shift
  sbatch -w "$NODE" -p "$PARTITION" --gres=gpu:1 --time="$time_limit" --mem="$mem" \
    --cpus-per-task="$cpus" --job-name="$job_name" \
    --output="$PROJECT/logs/formal_runs/${job_name}_%j.out" \
    --error="$PROJECT/logs/formal_runs/${job_name}_%j.err" "$script"
}
case "${1:-all}" in
  tju_ncm_pop_ind)
    submit formal_tju_ncm_pop_ind 12:00:00 64G 8 "$PROJECT/analysis/formal_runs/run_tju_ncm_pop_ind.sh";;
  r4_occlusion)
    submit formal_r4_occlusion 24:00:00 128G 8 "$PROJECT/analysis/formal_runs/run_r4_occlusion.sh";;
  *) echo "unknown target $1" >&2; exit 1;;
esac
