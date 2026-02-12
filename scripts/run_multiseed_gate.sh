#!/usr/bin/env bash
set -euo pipefail

# Multi-seed runner for gate_reference.py.
#
# You pass a *base* command (without --seed and without --results_dir).
# This script will append:
#   --seed <s> --results_dir <results_base>/seed_<s>
#
# Usage:
#   bash scripts/run_multiseed_gate.sh "1,2,3" <results_base> python gate_reference.py [args...]
#
# Example:
#   bash scripts/run_multiseed_gate.sh "1,2,3" ./results/jitter_multiseed \
#     python gate_reference.py \
#       --data_root /path/to/BCICIV_2a_mat \
#       --train_ref_modes native,car,laplacian,bipolar,gs,median \
#       --test_ref_modes native,car,laplacian,bipolar,gs,median \
#       --jitter_train_refs \
#       --jitter_ref_modes native,car,laplacian,bipolar,gs,median

SEEDS_CSV=${1:-"1"}
RESULTS_BASE=${2:?"Provide a results_base directory as arg2"}
shift 2

if [[ $# -lt 1 ]]; then
  echo "ERROR: provide the base command (e.g., python gate_reference.py ...)." >&2
  exit 1
fi

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
mkdir -p "$RESULTS_BASE"

for s in "${SEEDS[@]}"; do
  out_dir="$RESULTS_BASE/seed_${s}"
  mkdir -p "$out_dir"
  echo ""
  echo "===== SEED=$s  results_dir=$out_dir ====="
  "$@" --seed "$s" --results_dir "$out_dir"
done
