# Reference-Invariant MI-Net

This repo is a reproducible benchmark and baseline suite for **reference-mismatch robustness** in EEG motor-imagery decoding.

Core idea: a model trained under one valid referencing scheme (native, CAR, re-reference to a recorded channel, Laplacian) can **collapse** when evaluated on another scheme, even though the underlying brain signal is unchanged. The repo provides:

* a controlled *train-by-test reference matrix* protocol
* strong supervised baselines (concat-mix and reference-jitter)
* SSL pretraining + fine-tuning baselines, including **reference-views** (positives are the same trial under two reference transforms)

The backbone is deliberately **not** the contribution: we use a strong, known MI model (ATCNet) so the paper can focus on the failure mode and the fixes.

## Quick start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Point `DATA_ROOT` to the BCI Competition IV-2a data (as expected by the loader in `src/datamodules/bci2a.py`).

3. Run the gate benchmark

```bash
python gate_reference.py \
  --data_root "$DATA_ROOT" \
  --results_dir ./results/gate_EAoff_train_native \
  --train_ref_modes native \
  --test_ref_modes native,car,ref,laplacian \
  --no-ea \
  --epochs 200 \
  --seed 1
```

Results are written to:

* `.../gate_reference_summary.json` (args, averaged summary, per-subject breakdown)
* per-subject folders containing `splits.json`, `best.weights.h5`, `last.weights.h5`, and `weights_meta.json`

## Main scripts

* `gate_reference.py`:
  * single-reference training
  * concat-mix training (`--mix_train_refs`)
  * reference-jitter training (`--jitter_train_refs`, `--jitter_ref_modes`)
  * evaluates across `--test_ref_modes`

* `train_ssl.py`:
  * LOSO SSL pretraining (`--loso`)
  * view modes:
    * `aug` (standard augmentations)
    * `ref+aug` (positive pair is the same trial under two reference transforms, optionally with augmentations)
