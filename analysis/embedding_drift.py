"""Measure how model embeddings move under re-referencing.

This is a *post-hoc* diagnostic: given a trained model checkpoint, compute
representations for the same underlying trials under different reference modes,
then quantify stability.

Why this matters
----------------
If your core story is "reference mismatch causes representation failure",
showing embedding drift is the shortest path from accuracy tables to a
mechanistic claim:

* In-family refs should yield smaller drift than across-family refs.
* Unseen refs (LOO-ref) should produce large drift, especially native.

The script uses the model's `ssl_feat` head, which exists in your ATCNet
implementation when `return_ssl_feat=True`.
"""

from __future__ import annotations

import os
import sys

# Ensure repo root (the directory containing 'src') is on sys.path,
# even if this script is executed from a nested path after zip extraction.
_HERE = os.path.abspath(os.path.dirname(__file__))
_cand = _HERE
for _ in range(6):
    if os.path.isdir(os.path.join(_cand, "src")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
    _cand = os.path.dirname(_cand)

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.datamodules.bci2a import BCI2A_CH_NAMES, load_subject_dependent
from src.datamodules.channels import (
    name_to_index,
    neighbors_to_index_list,
    parse_keep_channels,
)
from src.datamodules.transforms import apply_reference
from src.models.model import build_atcnet


def _parse_modes(s: str) -> List[str]:
    return [m.strip() for m in (s or "").split(",") if m.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # [N,D] -> [N]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(an * bn, axis=1)


def _std_instance(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=-1, keepdims=True)
    sd = np.std(X, axis=-1, keepdims=True) + 1e-12
    return (X - mu) / sd


def _std_trainfit(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(Xtr, axis=(0, 2), keepdims=True)
    sd = np.std(Xtr, axis=(0, 2), keepdims=True) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _resolve_weights(args) -> str:
    if args.weights_path:
        return args.weights_path
    if not args.results_dir:
        raise ValueError("Provide --weights_path or --results_dir")
    fname = "best.weights.h5" if args.which == "best" else "last.weights.h5"

    # 0) If user points results_dir directly at a training folder, accept it.
    direct = os.path.join(args.results_dir, fname)
    if os.path.exists(direct):
        return direct

    # 1) SSL-style naming: <results_dir>/SUBJ_XX/{best,last}.weights.h5
    ssl_style = os.path.join(args.results_dir, f"SUBJ_{args.subject:02d}", fname)
    if os.path.exists(ssl_style):
        return ssl_style

    # 2) gate_reference-style naming: <results_dir>/sub_XX/train_<cond>/{best,last}.weights.h5
    sub_root = os.path.join(args.results_dir, f"sub_{args.subject:02d}")
    if args.cond:
        gate_style = os.path.join(sub_root, f"train_{args.cond}", fname)
        if os.path.exists(gate_style):
            return gate_style

    # 3) Auto-detect if there's exactly one train_* folder with weights
    if os.path.isdir(sub_root):
        cand_train = []
        for d in sorted(os.listdir(sub_root)):
            if not d.startswith("train_"):
                continue
            p = os.path.join(sub_root, d, fname)
            if os.path.exists(p):
                cand_train.append(p)
        if len(cand_train) == 1:
            return cand_train[0]
        if len(cand_train) > 1:
            raise FileNotFoundError(
                "Multiple candidate checkpoints found under "
                f"{sub_root}. Provide --cond (e.g. native, car, laplacian, ...) or use --weights_path.\n"
                "Candidates:\n  " + "\n  ".join(cand_train)
            )

    raise FileNotFoundError(
        "Could not resolve weights from --results_dir. Provide --weights_path, or point --results_dir at a train_* folder, "
        "or provide --cond for gate_reference outputs. Tried:\n"
        f"  {direct}\n  {ssl_style}\n  {os.path.join(sub_root, 'train_<cond>', fname)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument(
        "--cond",
        type=str,
        default="",
        help="Training condition name for gate_reference outputs (e.g. native, car, laplacian). Only needed if --results_dir contains multiple train_* folders.",
    )
    ap.add_argument(
        "--ref_modes",
        type=str,
        default="native,car,laplacian,bipolar,gs,median",
        help="Comma-separated list",
    )
    ap.add_argument(
        "--standardize_mode",
        type=str,
        default="instance",
        choices=["none", "instance", "train"],
        help="For embedding drift, 'instance' is usually the cleanest.",
    )
    ap.add_argument("--keep_channels", type=str, default=None)
    ap.add_argument("--ref_channel", type=str, default="Cz")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--max_trials", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--weights_path", type=str, default=None)
    ap.add_argument("--results_dir", type=str, default=None)
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])

    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    modes = _parse_modes(args.ref_modes)
    if not modes:
        raise ValueError("Empty --ref_modes")

    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    if keep_idx is None:
        keep_names = list(BCI2A_CH_NAMES)
    else:
        keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx]

    # ref_idx is needed for modes like 'ref' and for bipolar root selection
    name_to_i = name_to_index(keep_names)
    if args.ref_channel not in name_to_i:
        raise ValueError(f"ref_channel '{args.ref_channel}' not in channels: {keep_names}")
    ref_idx = int(name_to_i[args.ref_channel])

    lap_nb = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=keep_names, sort_by_distance=True)

    # Load native base once (unstandardized). We'll apply reference + standardization ourselves.
    (Xtr0, _ytr0), (Xte0, _yte0) = load_subject_dependent(
    args.data_root,
    args.subject,
    ea=False,
    standardize=False,
    ref_mode="native",
    keep_channels=args.keep_channels,
    ref_channel=args.ref_channel,
    laplacian=True,
)

    if args.split == "train":
        base = Xtr0
    else:
        base = Xte0

    if args.max_trials and base.shape[0] > args.max_trials:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(base.shape[0], size=args.max_trials, replace=False)
        base = base[sel]

    # Apply each reference to the same underlying trials
    X_by_mode: Dict[str, np.ndarray] = {}
    for mode in modes:
        X_by_mode[mode] = apply_reference(base, mode=mode, ref_idx=ref_idx, lap_neighbors=lap_nb)

    # Standardization
    if args.standardize_mode == "instance":
        X_by_mode = {m: _std_instance(X) for m, X in X_by_mode.items()}
    elif args.standardize_mode == "train":
        # Fit stats per mode on the *train* split for that mode, then apply to chosen split.
        # This matches your typical evaluation but can confound cross-mode comparisons.
        for mode in modes:
            Xtr_m = apply_reference(Xtr0, mode=mode, ref_idx=ref_idx, lap_neighbors=lap_nb)
            _, X_m = _std_trainfit(Xtr_m, X_by_mode[mode])
            X_by_mode[mode] = X_m

    # Build embedding model
    n_ch = X_by_mode[modes[0]].shape[1]
    model = build_atcnet(
        n_chans=n_ch,
        n_classes=4,
        return_ssl_feat=True,
        seed=args.seed,
    )

    weights_path = _resolve_weights(args)
    model.load_weights(weights_path)

    # Create a submodel that outputs ssl_feat only
    # build_atcnet returns [logits, ssl_feat] when return_ssl_feat=True.
    emb_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs[1])

    # Compute embeddings
    emb_by_mode: Dict[str, np.ndarray] = {}
    for mode in modes:
        X = X_by_mode[mode]
        X_in = X[:, None, :, :]  # [N,1,C,T]
        emb = emb_model.predict(X_in, batch_size=args.batch_size, verbose=0)
        emb_by_mode[mode] = np.asarray(emb)

    # Pairwise similarity statistics
    sim: Dict[str, Dict[str, Dict[str, float]]] = {a: {} for a in modes}
    for a in modes:
        for b in modes:
            s = _cosine_sim(emb_by_mode[a], emb_by_mode[b])
            sim[a][b] = {"mean": float(np.mean(s)), "std": float(np.std(s))}

    # Save
    np.savez_compressed(
        os.path.join(args.out_dir, "embeddings.npz"),
        **{f"emb_{m}": emb_by_mode[m] for m in modes},
    )
    with open(os.path.join(args.out_dir, "similarity.json"), "w") as f:
        json.dump(
            {
                "weights": weights_path,
                "subject": args.subject,
                "split": args.split,
                "standardize_mode": args.standardize_mode,
                "keep_channels": keep_names,
                "ref_modes": modes,
                "pairwise_cosine_similarity": sim,
            },
            f,
            indent=2,
        )

    print(f"Wrote: {os.path.join(args.out_dir, 'similarity.json')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'embeddings.npz')}")


if __name__ == "__main__":
    main()