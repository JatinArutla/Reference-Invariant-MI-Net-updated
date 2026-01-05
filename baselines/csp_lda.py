#!/usr/bin/env python
"""Classical baseline: CSP + LDA under reference shifts.

This is intentionally minimal so reviewers can reproduce quickly.

Protocol matches the main benchmark (no LOSO):
  - For each subject: train on session T, test on session E.
  - Apply reference operator at train and/or test.

Outputs a train x test matrix averaged across subjects.

Dependencies: numpy, scipy, sklearn.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.datamodules.bci2a import load_bci2a_session
from src.datamodules.transforms import standardize_pair


def _cov_normalized(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Trial covariance normalized by trace.

    X: [C,T]
    """
    C = X @ X.T
    tr = np.trace(C)
    return C / (tr + eps)


def _mean_cov(X: np.ndarray) -> np.ndarray:
    """Mean normalized covariance over trials.

    X: [N,C,T]
    """
    covs = np.array([_cov_normalized(x) for x in X], dtype=np.float64)
    return covs.mean(axis=0)


def _csp_ovr_filters(X: np.ndarray, y: np.ndarray, *, m: int = 2, eps: float = 1e-8) -> np.ndarray:
    """Compute one-vs-rest CSP filters for multi-class problems.

    Returns W with shape [C, F] where F = n_classes * 2*m.
    """
    classes = np.unique(y)
    C = X.shape[1]
    filters: List[np.ndarray] = []

    for k in classes:
        Xk = X[y == k]
        Xr = X[y != k]
        if len(Xk) < 2 or len(Xr) < 2:
            # Not enough data; fall back to identity slice.
            continue

        Rk = _mean_cov(Xk)
        Rr = _mean_cov(Xr)
        R = Rk + Rr

        # Solve generalized eigenproblem Rk v = lambda R v
        # eigh gives sorted ascending by default.
        w, v = eigh(Rk + eps * np.eye(C), R + eps * np.eye(C))
        idx = np.argsort(w)
        v = v[:, idx]

        # Take m smallest and m largest eigenvectors.
        Wk = np.concatenate([v[:, :m], v[:, -m:]], axis=1)  # [C, 2m]
        filters.append(Wk)

    if not filters:
        return np.eye(C, dtype=np.float32)

    W = np.concatenate(filters, axis=1).astype(np.float32)  # [C, F]
    return W


def _features_logvar(X: np.ndarray, W: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """CSP log-variance features.

    X: [N,C,T]
    W: [C,F]
    Returns: [N,F]
    """
    # Project: [N,F,T]
    Z = np.einsum("nct,cf->nft", X, W)
    var = np.var(Z, axis=2) + eps
    return np.log(var).astype(np.float32)


@dataclass
class Result:
    acc: float


def eval_subject(
    data_root: str,
    subject: int,
    *,
    train_mode: str,
    test_mode: str,
    t1_sec: float,
    t2_sec: float,
    keep_channels: str | None,
    ref_channel: str,
    laplacian: bool,
    csp_m: int,
    seed: int,
) -> Result:
    Xtr, ytr = load_bci2a_session(
        data_root,
        subject,
        training=True,
        all_trials=True,
        t1_sec=t1_sec,
        t2_sec=t2_sec,
        ref_mode=train_mode,
        keep_channels=keep_channels,
        ref_channel=ref_channel,
        laplacian=laplacian,
    )
    Xte, yte = load_bci2a_session(
        data_root,
        subject,
        training=False,
        all_trials=True,
        t1_sec=t1_sec,
        t2_sec=t2_sec,
        ref_mode=test_mode,
        keep_channels=keep_channels,
        ref_channel=ref_channel,
        laplacian=laplacian,
    )

    # Match deep-learning preprocessing: standardize using train stats.
    Xtr, Xte = standardize_pair(Xtr, Xte)

    W = _csp_ovr_filters(Xtr, ytr, m=csp_m)
    Ftr = _features_logvar(Xtr, W)
    Fte = _features_logvar(Xte, W)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(Ftr, ytr)
    pred = clf.predict(Fte)
    acc = float((pred == yte).mean() * 100.0)
    return Result(acc=acc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_ref_modes", type=str, default="native")
    p.add_argument("--test_ref_modes", type=str, default="native,car,laplacian,bipolar,gs,median")
    p.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9")
    p.add_argument("--t1_sec", type=float, default=2.0)
    p.add_argument("--t2_sec", type=float, default=6.0)
    p.add_argument("--keep_channels", type=str, default="")
    p.add_argument("--ref_channel", type=str, default="Cz")
    p.add_argument("--laplacian", action="store_true")
    p.add_argument("--csp_m", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    train_modes = [m.strip() for m in args.train_ref_modes.split(",") if m.strip()]
    test_modes = [m.strip() for m in args.test_ref_modes.split(",") if m.strip()]
    subs = [int(s) for s in args.subjects.split(",") if s.strip()]
    keep = args.keep_channels.strip() or None

    mat = np.zeros((len(train_modes), len(test_modes)), dtype=np.float64)

    for i, trm in enumerate(train_modes):
        for j, tem in enumerate(test_modes):
            accs = []
            for sub in subs:
                r = eval_subject(
                    args.data_root,
                    sub,
                    train_mode=trm,
                    test_mode=tem,
                    t1_sec=args.t1_sec,
                    t2_sec=args.t2_sec,
                    keep_channels=keep,
                    ref_channel=args.ref_channel,
                    laplacian=args.laplacian,
                    csp_m=args.csp_m,
                    seed=args.seed,
                )
                accs.append(r.acc)
            mat[i, j] = float(np.mean(accs))

    # Pretty print
    header = "train\\test".ljust(12) + "".join([f"{m:>12}" for m in test_modes])
    print("Averaged accuracy (%) - CSP+LDA")
    print(header)
    print("-" * len(header))
    for i, trm in enumerate(train_modes):
        row = trm.ljust(12) + "".join([f"{mat[i, j]:12.2f}" for j in range(len(test_modes))])
        print(row)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, train_modes=train_modes, test_modes=test_modes, acc=mat)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
