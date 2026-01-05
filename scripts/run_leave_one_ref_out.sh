#!/usr/bin/env bash
set -euo pipefail

# Leave-one-reference-out evaluation.
#
# IMPORTANT: The base command you pass should NOT include any of:
#   --train_ref_modes, --test_ref_modes, --jitter_train_refs, --jitter_ref_modes, --results_dir
# This script will set them.
#
# Usage:
#   bash scripts/run_leave_one_ref_out.sh "native,car,laplacian,bipolar,gs,median" ./results/loo_ref \
#     python gate_reference.py --data_root /path/to/BCICIV_2a_mat --no-loso --no-ea

MODES_CSV=${1:?"Provide comma-separated modes as arg1"}
RESULTS_BASE=${2:?"Provide results_base directory as arg2"}
shift 2

if [[ $# -lt 1 ]]; then
  echo "ERROR: provide base command (e.g., python gate_reference.py --data_root ...)." >&2
  exit 1
fi

IFS=',' read -ra MODES <<< "$MODES_CSV"
mkdir -p "$RESULTS_BASE"

for held in "${MODES[@]}"; do
  # Build training mode list excluding the held-out mode.
  train_modes=()
  for m in "${MODES[@]}"; do
    if [[ "$m" != "$held" ]]; then
      train_modes+=("$m")
    fi
  done
  train_csv=$(IFS=','; echo "${train_modes[*]}")

  out_dir="$RESULTS_BASE/heldout_${held}"
  mkdir -p "$out_dir"
  echo ""
  echo "===== HELD-OUT=$held  train_modes=$train_csv  results_dir=$out_dir ====="

  "$@" \
    --results_dir "$out_dir" \
    --train_ref_modes "$train_csv" \
    --jitter_train_refs \
    --jitter_ref_modes "$train_csv" \
    --test_ref_modes "$held"
done
