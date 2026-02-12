"""Audit how referencing changes basic signal features.

This is deliberately *not* about improving accuracy.
It's about making your reference-mismatch story more mechanistic and reviewer-proof.

What it computes (per trial)
---------------------------
1) Spectral aperiodic slope (log-log PSD fit, 4-40 Hz)
2) Oscillatory "peak residual" strength in mu (8-13) and beta (13-30)
3) Bandpower (mu/beta) in selected channels (defaults are motor-region-ish)
4) Spatial gradient energy proxies (laplacian energy, bipolar energy)

It then summarizes each metric per reference mode and outputs pairwise
"standardized mean shift" distances between modes.

Run example
-----------
python analysis/feature_shift_audit.py \
  --data_root /kaggle/input/four-class-motor-imagery-bnci-001-2014 \
  --subject 1 \
  --ref_modes native,car,laplacian,bipolar,gs,median \
  --standardize_mode none \
  --keep_channels CANON_CHS_18 \
  --split test \
  --out_dir /kaggle/working/feature_audit_sub01

Notes
-----
* If you want the analysis to reflect *pure* referencing (not "train-fit" stats),
  prefer --standardize_mode none or instance.
* The slope/peak extraction here is a lightweight alternative to FOOOF: it fits a
  line to log10 PSD vs log10 f and uses residual peaks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

from src.datamodules.bci2a import load_subject_dependent
from src.datamodules.channels import (
    BCI2A_CH_NAMES,
    parse_keep_channels,
    neighbors_to_index_list,
    name_to_index,
)
from src.datamodules.transforms import apply_reference


def _parse_modes(s: str) -> List[str]:
    return [m.strip() for m in s.split(",") if m.strip()]


def _parse_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out or None


def _welch_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    # x: [T]
    nperseg = min(256, x.shape[0])
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f, pxx


def _bandpower(f: np.ndarray, pxx: np.ndarray, fmin: float, fmax: float) -> float:
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return float("nan")
    return float(np.trapz(pxx[m], f[m]))


def _slope_and_residual_peaks(
    f: np.ndarray,
    pxx: np.ndarray,
    fit_lo: float = 4.0,
    fit_hi: float = 40.0,
    peak_bands: Tuple[Tuple[float, float], Tuple[float, float]] = ((8.0, 13.0), (13.0, 30.0)),
) -> Tuple[float, float, float]:
    """Return (slope, mu_peak_resid, beta_peak_resid).

    Fit line in log10 space: log10(P) = a + b log10(f).
    We return b (slope) and maximum positive residual in each band.
    """

    m = (f >= fit_lo) & (f <= fit_hi)
    f2 = f[m]
    p2 = pxx[m]
    if f2.size < 5:
        return float("nan"), float("nan"), float("nan")

    # avoid log(0)
    y = np.log10(p2 + 1e-12)
    x = np.log10(f2 + 1e-12)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    yhat = a + b * x
    resid = y - yhat

    def max_resid(lo: float, hi: float) -> float:
        mm = (f2 >= lo) & (f2 <= hi)
        if not np.any(mm):
            return float("nan")
        return float(np.max(resid[mm]))

    mu_r = max_resid(*peak_bands[0])
    beta_r = max_resid(*peak_bands[1])
    return float(b), mu_r, beta_r


def _laplacian_energy(X: np.ndarray, neighbors: List[List[int]]) -> float:
    """Mean squared laplacian output, averaged over channels and time."""
    Y = apply_reference(X[None, ...], ref_mode="laplacian", laplacian_neighbors=neighbors)[0]
    return float(np.mean(Y**2))


def _bipolar_energy(X: np.ndarray, neighbors: List[List[int]]) -> float:
    Y = apply_reference(X[None, ...], ref_mode="bipolar", laplacian_neighbors=neighbors)[0]
    return float(np.mean(Y**2))


@dataclass
class TrialFeatures:
    slope: float
    mu_peak_resid: float
    beta_peak_resid: float
    mu_power: float
    beta_power: float
    laplacian_energy: float
    bipolar_energy: float


def _standardized_mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    """|| (mu_a - mu_b) / pooled_std ||_2 over feature dimensions."""
    mu_a = np.nanmean(a, axis=0)
    mu_b = np.nanmean(b, axis=0)
    sd_a = np.nanstd(a, axis=0)
    sd_b = np.nanstd(b, axis=0)
    pooled = np.sqrt(0.5 * (sd_a**2 + sd_b**2)) + 1e-12
    z = (mu_a - mu_b) / pooled
    return float(np.linalg.norm(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument(
        "--ref_modes",
        type=str,
        default="native,car,laplacian,bipolar,gs,median",
        help="Comma-separated list.",
    )
    ap.add_argument(
        "--standardize_mode",
        type=str,
        default="none",
        choices=["none", "instance", "train"],
        help="Use 'none' or 'instance' to keep this audit about referencing, not train-fit stats.",
    )
    ap.add_argument("--keep_channels", type=str, default=None)
    ap.add_argument(
        "--channels",
        type=str,
        default="FC3,FC4,C3,Cz,C4,CP3,CP4",
        help="Channels to compute PSD features on (comma-separated). Must exist after keep_channels.",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "both"])
    ap.add_argument("--max_trials", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    modes = _parse_modes(args.ref_modes)
    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    if keep_idx is None:
        keep_names = list(BCI2A_CH_NAMES)
    else:
        keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx]

    # neighbor lists in kept montage
    lap_nb = neighbors_to_index_list(BCI2A_CH_NAMES, keep_names, mode="laplacian")
    bip_nb = neighbors_to_index_list(BCI2A_CH_NAMES, keep_names, mode="bipolar_nn")

    # channel indices for PSD metrics
    ch_list = _parse_list(args.channels) or []
    ch_idx: List[int] = []
    for nm in ch_list:
        try:
            ch_idx.append(name_to_index(nm, keep_names))
        except Exception as e:
            raise ValueError(f"Channel '{nm}' not found in kept montage. keep_channels={args.keep_channels}") from e

    # Load *native, unstandardized* once so every mode starts from identical trials.
    # We then apply reference + (optional) standardization ourselves in a controlled way.
    Xtr0, _, Xte0, _ = load_subject_dependent(
        data_root=args.data_root,
        subject=args.subject,
        ref_mode="native",
        standardize_mode="none",
        laplacian_neighbors=lap_nb,
        keep_idx=keep_idx,
        seed=args.seed,
    )

    # Optional subsampling indices are chosen once and reused across modes.
    if args.split == "train":
        base_train = Xtr0
        base_test = None
    elif args.split == "test":
        base_train = Xtr0
        base_test = Xte0
    else:
        base_train = Xtr0
        base_test = Xte0

    sel_train = None
    sel_test = None
    if args.max_trials:
        rng = np.random.default_rng(args.seed)
        if base_train is not None and base_train.shape[0] > args.max_trials:
            sel_train = rng.choice(base_train.shape[0], size=args.max_trials, replace=False)
        if base_test is not None and base_test.shape[0] > args.max_trials:
            sel_test = rng.choice(base_test.shape[0], size=args.max_trials, replace=False)

    # dataset sampling rate in this repo is fixed for IV-2a
    fs = 250.0

    results: Dict[str, Dict[str, float]] = {}
    per_mode_rows: Dict[str, List[List[float]]] = {}

    def _std_instance(X: np.ndarray) -> np.ndarray:
        mu = np.mean(X, axis=-1, keepdims=True)
        sd = np.std(X, axis=-1, keepdims=True) + 1e-12
        return (X - mu) / sd

    def _std_trainfit(Xtr: np.ndarray, Xte: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Fit per-channel mean/std over (trials,time)
        mu = np.mean(Xtr, axis=(0, 2), keepdims=True)
        sd = np.std(Xtr, axis=(0, 2), keepdims=True) + 1e-12
        Xtr2 = (Xtr - mu) / sd
        Xte2 = None if Xte is None else (Xte - mu) / sd
        return Xtr2, Xte2

    for mode in modes:
        # Apply reference transform to identical underlying trials.
        Xtr_m = apply_reference(base_train, ref_mode=mode, laplacian_neighbors=lap_nb)
        Xte_m = None if base_test is None else apply_reference(base_test, ref_mode=mode, laplacian_neighbors=lap_nb)

        if args.standardize_mode == "instance":
            Xtr_m = _std_instance(Xtr_m)
            if Xte_m is not None:
                Xte_m = _std_instance(Xte_m)
        elif args.standardize_mode == "train":
            Xtr_m, Xte_m = _std_trainfit(Xtr_m, Xte_m)

        if args.split == "train":
            X = Xtr_m
            if sel_train is not None:
                X = X[sel_train]
        elif args.split == "test":
            if Xte_m is None:
                raise RuntimeError("internal: Xte_m is None")
            X = Xte_m
            if sel_test is not None:
                X = X[sel_test]
        else:
            if Xte_m is None:
                raise RuntimeError("internal: Xte_m is None")
            Xtr_s = Xtr_m if sel_train is None else Xtr_m[sel_train]
            Xte_s = Xte_m if sel_test is None else Xte_m[sel_test]
            X = np.concatenate([Xtr_s, Xte_s], axis=0)

        rows: List[List[float]] = []
        for n in range(X.shape[0]):
            Xi = X[n]  # [C,T]

            # PSD metrics: average across selected channels
            slopes = []
            mu_peaks = []
            beta_peaks = []
            mu_pows = []
            beta_pows = []

            for ci in ch_idx:
                f, pxx = _welch_psd(Xi[ci], fs)
                slope, mu_r, beta_r = _slope_and_residual_peaks(f, pxx)
                slopes.append(slope)
                mu_peaks.append(mu_r)
                beta_peaks.append(beta_r)
                mu_pows.append(_bandpower(f, pxx, 8.0, 13.0))
                beta_pows.append(_bandpower(f, pxx, 13.0, 30.0))

            slope = float(np.nanmean(slopes))
            mu_peak = float(np.nanmean(mu_peaks))
            beta_peak = float(np.nanmean(beta_peaks))
            mu_pow = float(np.nanmean(mu_pows))
            beta_pow = float(np.nanmean(beta_pows))

            lap_e = _laplacian_energy(Xi, lap_nb)
            bip_e = _bipolar_energy(Xi, bip_nb)

            tf = TrialFeatures(
                slope=slope,
                mu_peak_resid=mu_peak,
                beta_peak_resid=beta_peak,
                mu_power=mu_pow,
                beta_power=beta_pow,
                laplacian_energy=lap_e,
                bipolar_energy=bip_e,
            )
            rows.append(list(asdict(tf).values()))

        feats = np.asarray(rows, dtype=np.float64)
        per_mode_rows[mode] = rows

        # summary
        summary = {}
        keys = list(asdict(TrialFeatures(0, 0, 0, 0, 0, 0, 0)).keys())
        for k_i, k in enumerate(keys):
            summary[f"{k}_mean"] = float(np.nanmean(feats[:, k_i]))
            summary[f"{k}_std"] = float(np.nanstd(feats[:, k_i]))
        results[mode] = summary

        # Save per-mode feature arrays for deeper analysis later.
        np.save(os.path.join(args.out_dir, f"features_{mode}.npy"), feats)

    # Pairwise shift distances
    keys = list(per_mode_rows.keys())
    shift = {a: {} for a in keys}
    for a in keys:
        A = np.asarray(per_mode_rows[a], dtype=np.float64)
        for b in keys:
            B = np.asarray(per_mode_rows[b], dtype=np.float64)
            shift[a][b] = _standardized_mean_shift(A, B)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "keep_channels": keep_names,
                "channels_for_psd": ch_list,
                "ref_modes": keys,
                "split": args.split,
                "summary": results,
                "pairwise_standardized_mean_shift": shift,
            },
            f,
            indent=2,
        )

    # Also write a compact CSV
    csv_path = os.path.join(args.out_dir, "summary.csv")
    headers = ["mode"] + sorted(next(iter(results.values())).keys())
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(headers)
        for mode in keys:
            wcsv.writerow([mode] + [results[mode][h] for h in headers[1:]])

    print(f"Wrote: {os.path.join(args.out_dir, 'summary.json')}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
