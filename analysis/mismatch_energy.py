#!/usr/bin/env python
"""Mismatch energy decomposition for reference operators.

This is a data-only mechanistic analysis.

For the same underlying trials x (native-as-loaded, before any standardization),
we apply each reference transform R_k and compare pairs:

  Δ_{i→j}(x) = R_j x - R_i x

We report:
  - Total mismatch energy:   E[||Δ||_F^2]
  - Fraction in common-mode: E[||Δ_cm||_F^2] / E[||Δ||_F^2]
    where Δ_cm is the per-timepoint channel-mean replicated across channels.
  - (Optional) normalize by E[||R_i x||_F^2] to make scales comparable.

Why it matters:
  - Global references differ mainly by common-mode subspace changes.
  - Local derivative-like references (laplacian/bipolar) differ in other spatial subspaces.

Outputs:
  - <out_dir>/mismatch_energy.json (aggregate across subjects)
  - <out_dir>/mismatch_energy_subXX.npz (per subject matrices)
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
import json
from typing import Dict, List, Tuple

import numpy as np

from src.datamodules.bci2a import load_subject_dependent
from src.datamodules.channels import (
    BCI2A_CH_NAMES,
    name_to_index,
    neighbors_to_index_list,
    parse_keep_channels,
)
from src.datamodules.transforms import apply_reference


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_subjects(s: str, n_sub: int) -> List[int]:
    s2 = (s or "").strip().lower()
    if s2 in ("all", "*"):
        return list(range(1, int(n_sub) + 1))
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("--subjects must be non-empty (or use 'all')")
    return out


def _energy(x: np.ndarray) -> float:
    return float(np.sum(x.astype(np.float64) ** 2))


def _mismatch_stats(Xi: np.ndarray, Xj: np.ndarray, *, eps: float = 1e-12) -> Tuple[float, float]:
    """Return (total_energy, common_mode_fraction)."""
    d = (Xj - Xi).astype(np.float64)
    # common-mode: per-timepoint channel-mean replicated across channels
    d_cm = d.mean(axis=1, keepdims=True)
    e = float(np.sum(d * d))
    e_cm = float(np.sum(d_cm * d_cm))
    frac = float(e_cm / (e + eps))
    return e, frac


def _print_table(mat: np.ndarray, modes: List[str], title: str, *, fmt: str = "{:.3e}") -> None:
    print("\n" + title)
    header = "i\\j".ljust(12) + "".join([m.ljust(14) for m in modes])
    print(header)
    print("-" * len(header))
    for i, mi in enumerate(modes):
        row = mi.ljust(12)
        for j in range(len(modes)):
            row += (fmt.format(mat[i, j])).ljust(14)
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser("Mismatch energy decomposition")

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--subjects", type=str, default="1", help="Comma-separated subject ids or 'all'")
    ap.add_argument("--n_sub", type=int, default=9)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])

    ap.add_argument(
        "--ref_modes",
        type=str,
        default="native,car,laplacian,bipolar,gs,median",
        help="Comma-separated list of reference modes to compare",
    )
    ap.add_argument("--keep_channels", type=str, default="", help="Preset name or comma-separated channel names")
    ap.add_argument("--ref_channel", type=str, default="Cz")

    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize mismatch energy by E[||R_i x||^2] for each row (i).",
    )
    ap.add_argument(
        "--max_trials",
        type=int,
        default=0,
        help="If >0, subsample this many trials per subject (deterministic by seed)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    modes = _parse_list(args.ref_modes)
    if not modes:
        raise ValueError("Empty --ref_modes")

    subjects = _parse_subjects(args.subjects, args.n_sub)

    # Resolve channel subset + indices
    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx] if keep_idx is not None else list(BCI2A_CH_NAMES)
    n_ch = len(keep_names)

    name_to_i = name_to_index(keep_names)
    if args.ref_channel not in name_to_i:
        raise ValueError(f"ref_channel '{args.ref_channel}' not in channels: {keep_names}")
    ref_idx = int(name_to_i[args.ref_channel])
    lap_nb = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=keep_names, sort_by_distance=False)

    # Aggregate across subjects
    E_sum = np.zeros((len(modes), len(modes)), dtype=np.float64)
    CM_sum = np.zeros((len(modes), len(modes)), dtype=np.float64)
    ROW_norm = np.zeros((len(modes),), dtype=np.float64)

    per_subject: Dict[str, Dict[str, str]] = {}
    rng = np.random.default_rng(int(args.seed))

    for sub in subjects:
        (Xtr0, _ytr0), (Xte0, _yte0) = load_subject_dependent(
            args.data_root,
            int(sub),
            ea=False,
            standardize=False,
            ref_mode="native",
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=True,
        )

        base = Xtr0 if args.split == "train" else Xte0
        if args.max_trials and base.shape[0] > int(args.max_trials):
            sel = rng.choice(base.shape[0], size=int(args.max_trials), replace=False)
            sel = np.sort(sel)
            base = base[sel]

        X_by_mode: Dict[str, np.ndarray] = {}
        for m in modes:
            X_by_mode[m] = apply_reference(base, mode=m, ref_idx=ref_idx, lap_neighbors=lap_nb)
            if X_by_mode[m].shape[1] != n_ch:
                raise RuntimeError(f"Channel mismatch after apply_reference for mode={m}: got {X_by_mode[m].shape}")

        # Per-subject matrices
        E = np.zeros((len(modes), len(modes)), dtype=np.float64)
        CM = np.zeros((len(modes), len(modes)), dtype=np.float64)
        row_norm = np.zeros((len(modes),), dtype=np.float64)

        for i, mi in enumerate(modes):
            Xi = X_by_mode[mi]
            row_norm[i] = _energy(Xi)
            for j, mj in enumerate(modes):
                Xj = X_by_mode[mj]
                e, frac_cm = _mismatch_stats(Xi, Xj)
                E[i, j] = e
                CM[i, j] = frac_cm

        # Save per-subject
        out_npz = os.path.join(args.out_dir, f"mismatch_energy_sub{sub:02d}.npz")
        np.savez_compressed(
            out_npz,
            modes=np.array(modes, dtype=object),
            energy=E,
            common_mode_frac=CM,
            row_energy=row_norm,
            n_trials=int(base.shape[0]),
            keep_channels=np.array(keep_names, dtype=object),
        )
        per_subject[f"sub{sub:02d}"] = {"npz": os.path.abspath(out_npz), "n_trials": int(base.shape[0])}

        E_sum += E
        CM_sum += CM
        ROW_norm += row_norm

    # Subject-average
    S = float(len(subjects))
    E_mu = E_sum / max(S, 1.0)
    CM_mu = CM_sum / max(S, 1.0)
    ROW_mu = ROW_norm / max(S, 1.0)

    if args.normalize:
        # row-wise normalize mismatch energies
        E_mu = E_mu / (ROW_mu.reshape(-1, 1) + 1e-12)

    # Print quick tables
    _print_table(E_mu, modes, "Mean mismatch energy" + (" (row-normalized)" if args.normalize else ""))
    _print_table(CM_mu, modes, "Mean common-mode fraction of mismatch", fmt="{:.3f}")

    out_json = os.path.join(args.out_dir, "mismatch_energy.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "modes": modes,
                "keep_channels": keep_names,
                "subjects": subjects,
                "split": args.split,
                "mean_mismatch_energy": E_mu.tolist(),
                "mean_common_mode_fraction": CM_mu.tolist(),
                "mean_row_energy": ROW_mu.tolist(),
                "per_subject": per_subject,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
