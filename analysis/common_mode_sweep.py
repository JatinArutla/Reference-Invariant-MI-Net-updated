#!/usr/bin/env python
"""Common-mode injection sweep.

This script tests a concrete mechanistic hypothesis behind CAR/native directionality:

  - CAR projects signals into a subspace where the channel-mean (common-mode) is removed.
  - A model trained in that projected space can be fragile when a common-mode component
    is added back at test time.

We construct, per trial (C x T):

  x_car = CAR(x_native)
  m(t)  = mean_c x_native[c,t]  (the removed common-mode time-series)

  x(α) = x_car + α * 1 * m(t)

So α=0 is pure CAR input, α=1 reconstructs the native trial, and α>1 over-injects.

We then evaluate a *fixed* trained model on x(α) for a sweep of α.

Outputs:
  - out_dir/acc_vs_alpha.csv
  - out_dir/acc_vs_alpha.png (if matplotlib is available)
  - out_dir/meta.json

Typical usage (CAR-trained model):

  python analysis/common_mode_sweep.py \
    --data_root $DATA_ROOT \
    --results_dir $RESULTS_ROOT/full_train_car_stdtrain \
    --cond car --subject 1 --which best \
    --keep_channels canon --standardize_mode train \
    --out_dir $RESULTS_ROOT/mech/common_mode_sweep_car_model
"""

from __future__ import annotations

import os
import sys

# Ensure repo root (directory containing 'src') is on sys.path.
_HERE = os.path.abspath(os.path.dirname(__file__))
_cand = _HERE
for _ in range(6):
    if os.path.isdir(os.path.join(_cand, "src")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
    _cand = os.path.dirname(_cand)

import argparse
import csv
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.datamodules.bci2a import load_subject_dependent
from src.datamodules.channels import (
    BCI2A_CH_NAMES,
    name_to_index,
    neighbors_to_index_list,
    parse_keep_channels,
)
from src.datamodules.transforms import (
    apply_reference,
    fit_standardizer,
    apply_standardizer,
    standardize_instance,
)
from src.models.model import build_atcnet


def _parse_floats(s: str) -> List[float]:
    out: List[float] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _resolve_gate_weights(results_dir: str, subject: int, cond: str, which: str) -> str:
    fname = "best.weights.h5" if which == "best" else "last.weights.h5"
    cand_paths = [
        os.path.join(results_dir, f"sub_{int(subject):02d}", f"train_{cond}", fname),
        os.path.join(results_dir, f"SUBJ_{int(subject):02d}", f"train_{cond}", fname),
    ]
    for p in cand_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find weights. Tried:\n  " + "\n  ".join(cand_paths))


def _std_apply(mode: str, X: np.ndarray, mu_sd: Optional[Tuple[np.ndarray, np.ndarray]], robust: bool) -> np.ndarray:
    m = (mode or "none").lower()
    if m == "none":
        return X.astype(np.float32, copy=False)
    if m == "train":
        if mu_sd is None:
            raise ValueError("standardize_mode=train requires mu/sd")
        return apply_standardizer(X, *mu_sd).astype(np.float32, copy=False)
    if m == "instance":
        return standardize_instance(X, robust=robust).astype(np.float32, copy=False)
    raise ValueError(f"Unknown standardize_mode: {mode}")


def _acc_from_probs(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    y_hat = y_prob.argmax(-1).astype(int)
    return float((y_hat == y_true.astype(int)).mean())


def _maybe_plot(out_png: str, xs: List[float], ys: List[float], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, bbox_inches="tight", dpi=450)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser("Common-mode injection sweep")

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])

    # Weights
    ap.add_argument("--weights_path", type=str, default=None)
    ap.add_argument("--results_dir", type=str, default=None, help="gate_reference.py results_dir")
    ap.add_argument("--cond", type=str, default="car", help="train condition dir name: native/car/laplacian/bipolar/gs/median/jitter/mix")
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])

    # Channel / reference config
    ap.add_argument("--keep_channels", type=str, default="")
    ap.add_argument("--ref_channel", type=str, default="Cz")
    ap.add_argument("--standardize_mode", type=str, default="train", choices=["none", "instance", "train"])
    ap.add_argument("--instance_robust", action="store_true")
    ap.add_argument(
        "--fit_ref_mode",
        type=str,
        default="auto",
        help=(
            "Which ref mode to use to fit train statistics when standardize_mode=train. "
            "Use 'auto' to use --cond if it matches a single mode, else 'native'."
        ),
    )

    # Sweep config
    ap.add_argument(
        "--alphas",
        type=str,
        default="0,0.25,0.5,0.75,1.0,1.25",
        help="Comma-separated alpha values",
    )
    ap.add_argument(
        "--m_source",
        type=str,
        default="paired",
        choices=["paired", "random"],
        help="paired = use this trial's removed mean; random = use a mean sampled from another trial",
    )
    ap.add_argument(
        "--also_reverse",
        action="store_true",
        help=(
            "Also run the reverse sweep: native - alpha·mean (alpha=0 is native, alpha=1 matches CAR). "
            "This is useful for testing asymmetry with a native-trained model."
        ),
    )
    ap.add_argument("--max_trials", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    alphas = _parse_floats(args.alphas)
    if not alphas:
        raise ValueError("Empty --alphas")

    # Resolve channel subset
    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx] if keep_idx is not None else list(BCI2A_CH_NAMES)
    name_to_i = name_to_index(keep_names)
    if args.ref_channel not in name_to_i:
        raise ValueError(f"ref_channel '{args.ref_channel}' not in channels: {keep_names}")
    ref_idx = int(name_to_i[args.ref_channel])
    lap_nb = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=keep_names, sort_by_distance=False)

    # Load native once; apply CAR ourselves so trials are paired.
    (Xtr0, ytr0), (Xte0, yte0) = load_subject_dependent(
        args.data_root,
        int(args.subject),
        ea=False,
        standardize=False,
        ref_mode="native",
        keep_channels=args.keep_channels,
        ref_channel=args.ref_channel,
        laplacian=True,
    )

    X_base = Xtr0 if args.split == "train" else Xte0
    y_base = ytr0 if args.split == "train" else yte0

    rng = np.random.default_rng(int(args.seed))
    if args.max_trials and X_base.shape[0] > int(args.max_trials):
        sel = rng.choice(X_base.shape[0], size=int(args.max_trials), replace=False)
        sel = np.sort(sel)
        X_base = X_base[sel]
        y_base = y_base[sel]

    # Construct paired CAR and common-mode signal
    X_car = apply_reference(X_base, mode="car", ref_idx=ref_idx, lap_neighbors=lap_nb)
    # mean over channels → [N,T]
    m_paired = X_base.mean(axis=1)

    if args.m_source == "random":
        perm = rng.permutation(m_paired.shape[0])
        m_used = m_paired[perm]
    else:
        m_used = m_paired

    # Fit standardizer if needed
    mu_sd = None
    fit_ref = (args.fit_ref_mode or "auto").strip().lower()
    if args.standardize_mode == "train":
        if fit_ref == "auto":
            # If cond corresponds to a single concrete reference mode, use it; otherwise use native.
            if args.cond.lower() in ("native", "car", "laplacian", "bipolar", "gs", "median"):
                fit_ref = args.cond.lower()
            else:
                fit_ref = "native"
        Xtr_fit = apply_reference(Xtr0, mode=fit_ref, ref_idx=ref_idx, lap_neighbors=lap_nb)
        mu_sd = fit_standardizer(Xtr_fit)

    # Load model weights
    if args.weights_path:
        weights = args.weights_path
    else:
        if not args.results_dir:
            raise ValueError("Provide --weights_path or --results_dir")
        weights = _resolve_gate_weights(args.results_dir, args.subject, args.cond, args.which)

    n_ch = int(X_car.shape[1])
    model = build_atcnet(n_classes=4, in_chans=n_ch, in_samples=int(X_car.shape[2]), return_ssl_feat=False)
    model.load_weights(weights)

    rows: List[Dict[str, float]] = []

    # Sweep: CAR → add common-mode (reconstruct native at alpha=1)
    accs: List[float] = []
    for a in alphas:
        X_a = X_car + float(a) * m_used[:, None, :]
        X_a = _std_apply(args.standardize_mode, X_a, mu_sd, bool(args.instance_robust))
        y_prob = model.predict(X_a[:, None, :, :], batch_size=args.batch_size, verbose=0)
        acc = _acc_from_probs(y_prob, y_base)
        accs.append(acc)
        rows.append({"alpha": float(a), "accuracy": float(acc)})
        print(f"alpha={a:>6.3f}  acc={acc*100:6.2f}")

    # Write CSV + plot
    out_csv = os.path.join(args.out_dir, "acc_vs_alpha.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "accuracy"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_png = os.path.join(args.out_dir, "acc_vs_alpha.png")
    _maybe_plot(out_png, [r["alpha"] for r in rows], [r["accuracy"] for r in rows], title="Accuracy vs alpha (CAR + alpha·mean)")

    # Optional reverse: native → remove common-mode
    out_csv_rev = None
    out_png_rev = None
    if args.also_reverse:
        rows2: List[Dict[str, float]] = []
        for a in alphas:
            X_a = X_base - float(a) * m_used[:, None, :]
            X_a = _std_apply(args.standardize_mode, X_a, mu_sd, bool(args.instance_robust))
            y_prob = model.predict(X_a[:, None, :, :], batch_size=args.batch_size, verbose=0)
            acc = _acc_from_probs(y_prob, y_base)
            rows2.append({"alpha": float(a), "accuracy": float(acc)})
            print(f"[reverse] alpha={a:>6.3f}  acc={acc*100:6.2f}")

        out_csv_rev = os.path.join(args.out_dir, "acc_vs_alpha_reverse.csv")
        with open(out_csv_rev, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["alpha", "accuracy"])
            w.writeheader()
            for r in rows2:
                w.writerow(r)
        out_png_rev = os.path.join(args.out_dir, "acc_vs_alpha_reverse.png")
        _maybe_plot(out_png_rev, [r["alpha"] for r in rows2], [r["accuracy"] for r in rows2], title="Accuracy vs alpha (native - alpha·mean)")

    meta = {
        "weights": os.path.abspath(weights),
        "subject": int(args.subject),
        "split": args.split,
        "keep_channels": keep_names,
        "standardize_mode": args.standardize_mode,
        "fit_ref_mode": fit_ref if args.standardize_mode == "train" else None,
        "cond": args.cond,
        "which": args.which,
        "alphas": alphas,
        "m_source": args.m_source,
        "n_trials": int(X_base.shape[0]),
    }
    out_meta = os.path.join(args.out_dir, "meta.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_csv}")
    if os.path.exists(out_png):
        print(f"Saved: {out_png}")
    if out_csv_rev:
        print(f"Saved: {out_csv_rev}")
    if out_png_rev and os.path.exists(out_png_rev):
        print(f"Saved: {out_png_rev}")
    print(f"Saved: {out_meta}")


if __name__ == "__main__":
    main()
