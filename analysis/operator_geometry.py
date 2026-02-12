"""Analyze reference transforms as (approximate) channel-mixing operators.

Why this exists
--------------
Your main empirical finding is *directionality* and *family structure* in train→test
transfer across reference modes. One reviewer-grade way to ground that is to show
the transforms are not just arbitrary "domains" but have very specific linear
geometry (rank loss, projection-like behavior, subspace relationships).

This script computes:
- Exact linear operator matrices for strictly linear refs: CAR, laplacian, bipolar,
  randref (fixed weights).
- Optional *empirical best linear fit* for adaptive refs: GS, median.
  (These are not strictly linear as implemented in this repo.)

Run examples (from repo root)
-----------------------------
python analysis/operator_geometry.py \
  --ref_modes native,car,laplacian,bipolar,gs,median \
  --keep_channels CANON_CHS_18 \
  --out /kaggle/working/operator_geometry.json

To also estimate linear fits for GS/median on real data (slower):
python analysis/operator_geometry.py \
  --ref_modes car,laplacian,bipolar,gs,median \
  --approx_nonlinear_from_data \
  --data_root /kaggle/input/four-class-motor-imagery-bnci-001-2014 \
  --subject 1 \
  --out /kaggle/working/operator_geometry_with_fit.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import orth, subspace_angles

from src.datamodules.channels import (
    BCI2A_CH_NAMES,
    parse_keep_channels,
    neighbors_to_index_list,
)
from src.datamodules.bci2a import load_subject_dependent
from src.datamodules.transforms import apply_reference


def _parse_modes(s: str) -> List[str]:
    return [m.strip() for m in s.split(",") if m.strip()]


def _car_matrix(C: int) -> np.ndarray:
    I = np.eye(C, dtype=np.float64)
    ones = np.ones((C, C), dtype=np.float64) / float(C)
    return I - ones


def _laplacian_matrix(neighbors: List[List[int]]) -> np.ndarray:
    C = len(neighbors)
    A = np.eye(C, dtype=np.float64)
    for i in range(C):
        nb = neighbors[i]
        if not nb:
            continue
        w = -1.0 / float(len(nb))
        for j in nb:
            A[i, j] += w
    return A


def _bipolar_nn_matrix(neighbors: List[List[int]]) -> np.ndarray:
    """Dimension-preserving bipolar: each channel subtracts its closest neighbor."""
    C = len(neighbors)
    A = np.eye(C, dtype=np.float64)
    for i in range(C):
        nb = neighbors[i]
        if not nb:
            continue
        j0 = nb[0]
        if j0 == i:
            continue
        A[i, j0] -= 1.0
    return A


def _randref_matrix(weights: np.ndarray) -> np.ndarray:
    """randref in this repo: subtract a fixed weighted average from every channel."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    C = w.shape[0]
    I = np.eye(C, dtype=np.float64)
    return I - np.ones((C, 1), dtype=np.float64) @ w.reshape(1, C)


def _rank(A: np.ndarray, tol: float = 1e-9) -> int:
    return int(np.linalg.matrix_rank(A, tol=tol))


def _projection_error(A: np.ndarray) -> float:
    # ||A^2 - A||_F / ||A||_F
    num = np.linalg.norm(A @ A - A, ord="fro")
    den = np.linalg.norm(A, ord="fro") + 1e-12
    return float(num / den)


def _symmetry_error(A: np.ndarray) -> float:
    num = np.linalg.norm(A - A.T, ord="fro")
    den = np.linalg.norm(A, ord="fro") + 1e-12
    return float(num / den)


def _energy_stats(A: np.ndarray, n: int = 10_000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((A.shape[0], n)).astype(np.float64)
    y2 = np.sum((A @ X) ** 2, axis=0)
    x2 = np.sum(X**2, axis=0) + 1e-12
    r = y2 / x2
    return float(np.mean(r)), float(np.std(r))


def _rowspace_basis(A: np.ndarray) -> np.ndarray:
    # Row space of A == column space of A^T
    return orth(A.T)


def _subspace_angle_summary(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    QA = _rowspace_basis(A)
    QB = _rowspace_basis(B)
    if QA.size == 0 or QB.size == 0:
        return {"mean_deg": float("nan"), "max_deg": float("nan")}
    ang = subspace_angles(QA, QB)  # radians
    ang_deg = ang * (180.0 / np.pi)
    return {
        "mean_deg": float(np.mean(ang_deg)),
        "max_deg": float(np.max(ang_deg)),
        "k": int(len(ang_deg)),
    }


@dataclass
class OperatorStats:
    mode: str
    C: int
    rank: int
    null_dim: int
    proj_err: float
    sym_err: float
    energy_mean: float
    energy_std: float

    # For non-linear modes: how well a single linear operator fits (lower is better)
    linear_fit_relerr: Optional[float] = None


def _fit_linear_operator_from_data(
    *,
    data_root: str,
    subject: int,
    ref_mode: str,
    keep_idx: Optional[List[int]],
    laplacian_neighbors: List[List[int]],
    seed: int,
    max_trials: int,
) -> Tuple[np.ndarray, float]:
    """Fit A that minimizes ||Y - A X||_F over concatenated timepoints.

    This is purely diagnostic. For truly linear transforms, the fit recovers the
    exact operator (up to noise). For adaptive transforms (gs/median here), this
    quantifies "how non-linear" they behave on real data.
    """

    # Load both train + test, then subsample for speed.
    Xtr, ytr, Xte, yte = load_subject_dependent(
        data_root=data_root,
        subject=subject,
        ref_mode="native",
        standardize_mode="none",
        laplacian_neighbors=laplacian_neighbors,
        keep_idx=keep_idx,
        seed=seed,
    )
    X = np.concatenate([Xtr, Xte], axis=0)
    if max_trials and X.shape[0] > max_trials:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_trials, replace=False)
        X = X[idx]

    # Apply the reference transform to get Y.
    Y = apply_reference(X, ref_mode=ref_mode, laplacian_neighbors=laplacian_neighbors)

    # Stack timepoints: X_flat: [C, N*T]
    Xf = np.transpose(X, (1, 0, 2)).reshape(X.shape[1], -1).astype(np.float64)
    Yf = np.transpose(Y, (1, 0, 2)).reshape(Y.shape[1], -1).astype(np.float64)

    # Solve Y = A X in least squares: A = Y X^T (X X^T)^{-1}
    XXt = Xf @ Xf.T
    reg = 1e-6 * np.eye(XXt.shape[0])
    A = (Yf @ Xf.T) @ np.linalg.inv(XXt + reg)

    # Relative Frobenius error
    rel = float(np.linalg.norm(Yf - A @ Xf, ord="fro") / (np.linalg.norm(Yf, ord="fro") + 1e-12))
    return A, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref_modes",
        type=str,
        default="car,laplacian,bipolar,gs,median,randref",
        help="Comma-separated list of modes to analyze.",
    )
    ap.add_argument(
        "--keep_channels",
        type=str,
        default=None,
        help="Comma-separated channel names to keep, or preset CANON_CHS_18.",
    )
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    ap.add_argument(
        "--approx_nonlinear_from_data",
        action="store_true",
        help="Fit a best linear operator for gs/median using real data (slow).",
    )
    ap.add_argument("--data_root", type=str, default=None, help="Required if --approx_nonlinear_from_data")
    ap.add_argument("--subject", type=int, default=1, help="Subject for fitting gs/median operators")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_trials", type=int, default=400, help="Subsample for operator fitting")
    args = ap.parse_args()

    modes = _parse_modes(args.ref_modes)
    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    if keep_idx is None:
        keep_names = list(BCI2A_CH_NAMES)
    else:
        keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx]

    # Neighbors projected onto kept montage.
    lap_nb = neighbors_to_index_list(
        all_names=BCI2A_CH_NAMES,
        keep_names=keep_names,
        mode="laplacian",
    )
    bip_nb = neighbors_to_index_list(
        all_names=BCI2A_CH_NAMES,
        keep_names=keep_names,
        mode="bipolar_nn",
    )

    op_mats: Dict[str, np.ndarray] = {}
    stats: List[OperatorStats] = []

    C = len(keep_names)

    # Fixed randref weights (deterministic for diagnostics).
    rng = np.random.default_rng(0)
    w = rng.random(C).astype(np.float64)
    w = w / (np.sum(w) + 1e-12)

    for m in modes:
        m_key = m.strip().lower()
        A = None
        relerr = None

        if m_key in ("car",):
            A = _car_matrix(C)
        elif m_key in ("laplacian",):
            A = _laplacian_matrix(lap_nb)
        elif m_key in ("bipolar", "bipolar_nn"):
            A = _bipolar_nn_matrix(bip_nb)
        elif m_key in ("randref",):
            A = _randref_matrix(w)
        elif m_key in ("native",):
            # "native" is not a transform we apply; treat as identity operator.
            A = np.eye(C, dtype=np.float64)
        elif m_key in ("gs", "median"):
            if args.approx_nonlinear_from_data:
                if not args.data_root:
                    raise ValueError("--data_root is required for --approx_nonlinear_from_data")
                A, relerr = _fit_linear_operator_from_data(
                    data_root=args.data_root,
                    subject=args.subject,
                    ref_mode=m_key,
                    keep_idx=keep_idx,
                    laplacian_neighbors=lap_nb,
                    seed=args.seed,
                    max_trials=args.max_trials,
                )
            else:
                # Skip by default because there is no fixed operator.
                print(f"[skip] mode '{m_key}' is not strictly linear in this repo. Use --approx_nonlinear_from_data.")
                continue
        else:
            print(f"[skip] unsupported/unknown mode '{m_key}'")
            continue

        op_mats[m_key] = A
        r = _rank(A)
        mu, sd = _energy_stats(A)
        stats.append(
            OperatorStats(
                mode=m_key,
                C=C,
                rank=r,
                null_dim=C - r,
                proj_err=_projection_error(A),
                sym_err=_symmetry_error(A),
                energy_mean=mu,
                energy_std=sd,
                linear_fit_relerr=relerr,
            )
        )

    # Pairwise subspace angles for the operators we computed.
    pair_angles: Dict[str, Dict[str, Dict[str, float]]] = {}
    keys = list(op_mats.keys())
    for i, a in enumerate(keys):
        pair_angles[a] = {}
        for b in keys:
            pair_angles[a][b] = _subspace_angle_summary(op_mats[a], op_mats[b])

    # Pretty print a compact summary.
    print("\nOperator stats")
    print("mode\tC\trank\tnull\tproj_err\tsym_err\tEmean\tEstd\tlinfit_relerr")
    for s in stats:
        print(
            f"{s.mode}\t{s.C}\t{s.rank}\t{s.null_dim}\t"
            f"{s.proj_err:.3e}\t{s.sym_err:.3e}\t{s.energy_mean:.3f}\t{s.energy_std:.3f}\t"
            f"{'' if s.linear_fit_relerr is None else f'{s.linear_fit_relerr:.3e}'}"
        )

    out_obj = {
        "keep_channels": keep_names,
        "modes": keys,
        "operator_stats": [asdict(s) for s in stats],
        "pairwise_rowspace_angles": pair_angles,
        "notes": {
            "gs_median": "In this repo, gs and median are data-adaptive and not strictly linear. Use --approx_nonlinear_from_data to fit an empirical linear operator and quantify non-linearity.",
            "bipolar": "bipolar here is dimension-preserving nearest-neighbor difference (bipolar_nn).",
        },
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out_obj, f, indent=2)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
