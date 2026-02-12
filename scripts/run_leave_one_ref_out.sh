#!/usr/bin/env bash
set -euo pipefail

# Leave-one-reference-out (LOO-ref) runner.
#
# Usage (example):
#   bash scripts/run_leave_one_ref_out.sh "native,car,laplacian,bipolar,gs,median" \
#        /path/to/out \
#        python gate_reference.py --data_root ... --standardize_mode train --no-ea --laplacian --no-loso --epochs 200 --seed 1
#
# What this script enforces:
# - Base data is always loaded as 'native'.
# - The *held-out* mode appears ONLY in --test_ref_modes.
# - Checkpoint selection uses ONLY seen modes (--val_ref_modes excludes held-out).
# - Keras val_loss uses a single *seen* mode (--val_single_ref_mode), avoiding leakage into LR scheduling.

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <CSV_MODES> <OUT_ROOT> <command...>" >&2
  exit 2
fi

CSV_MODES="$1"
OUT_ROOT="$2"
shift 2

mkdir -p "$OUT_ROOT"

# Remaining args form the base command
CMD=("$@")

IFS=',' read -r -a MODES_ARR <<< "$CSV_MODES"

trim() {
  local s="$1"
  # shellcheck disable=SC2001
  echo "${s}" | sed 's/^ *//; s/ *$//'
}

join_csv_excluding() {
  local exclude="$1"
  local out=""
  for m in "${MODES_ARR[@]}"; do
    m="$(trim "$m")"
    if [[ -z "$m" ]]; then
      continue
    fi
    if [[ "$m" == "$exclude" ]]; then
      continue
    fi
    if [[ -z "$out" ]]; then
      out="$m"
    else
      out+="${out:+,}$m"
    fi
  done
  echo "$out"
}

first_mode_in_csv() {
  local csv="$1"
  IFS=',' read -r -a tmp <<< "$csv"
  echo "$(trim "${tmp[0]}")"
}

for held_raw in "${MODES_ARR[@]}"; do
  held="$(trim "$held_raw")"
  if [[ -z "$held" ]]; then
    continue
  fi

  train_csv="$(join_csv_excluding "$held")"
  if [[ -z "$train_csv" ]]; then
    echo "ERROR: after excluding '$held', no modes remain." >&2
    exit 3
  fi

  # Validation single mode must be seen to avoid leakage through ReduceLROnPlateau.
  # If held-out is 'native', pick the first remaining mode; otherwise use 'native'.
  if [[ "$held" == "native" ]]; then
    val_single="$(first_mode_in_csv "$train_csv")"
  else
    val_single="native"
  fi

  run_dir="$OUT_ROOT/heldout_${held}"
  mkdir -p "$run_dir"

  echo "\n===== LOO-REF: held-out=${held} | train/jitter modes=${train_csv} | val_single=${val_single} =====" >&2

  "${CMD[@]}" \
    --results_dir "$run_dir" \
    --train_ref_modes native \
    --jitter_train_refs \
    --jitter_ref_modes "$train_csv" \
    --test_ref_modes "$held" \
    --val_ref_modes "$train_csv" \
    --val_single_ref_mode "$val_single"
done
