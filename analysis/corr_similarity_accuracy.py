#!/usr/bin/env python
"""Correlate embedding similarity with generalization accuracy.

Inputs:
  - similarity.json from analysis/embedding_drift.py
  - gate_reference_summary.json from gate_reference.py

We correlate, for a chosen train mode m_train, across test modes m_test:

  x = cosine_similarity( z(m_train), z(m_test) )
  y = accuracy(train=m_train, test=m_test)

This is a quick sanity check: if representations are stable across operator swaps,
generalization should be better.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # rank with average ties
    def rank(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        for k, c in enumerate(cnt):
            if c > 1:
                idx = np.where(inv == k)[0]
                ranks[idx] = ranks[idx].mean()
        return ranks

    rx = rank(x)
    ry = rank(y)
    return _pearson(rx, ry)


def main() -> None:
    ap = argparse.ArgumentParser("Correlate similarity vs accuracy")
    ap.add_argument("--similarity_json", type=str, required=True)
    ap.add_argument("--gate_summary_json", type=str, required=True)
    ap.add_argument("--train_key", type=str, default="auto", help="Row key in gate summary: native/car/.../jitter/mix. Use 'auto' to infer from weights path.")
    ap.add_argument("--train_mode_for_similarity", type=str, default="auto", help="Mode name to use as anchor in similarity matrix. Use 'auto' to use --train_key.")
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    with open(args.similarity_json, "r") as f:
        sim_obj = json.load(f)
    with open(args.gate_summary_json, "r") as f:
        gate_obj = json.load(f)

    modes: List[str] = sim_obj["ref_modes"]
    sim = sim_obj["pairwise_cosine_similarity"]
    summary = gate_obj["summary"]

    train_key = args.train_key
    if train_key == "auto":
        # crude heuristic from weights path
        w = os.path.basename(os.path.dirname(sim_obj.get("weights", "")))
        # expects 'train_<key>'
        if w.startswith("train_"):
            train_key = w.split("train_", 1)[1]
        else:
            train_key = modes[0]

    train_mode = args.train_mode_for_similarity
    if train_mode == "auto":
        train_mode = train_key

    if train_key not in summary:
        raise KeyError(f"train_key '{train_key}' not found in gate summary keys: {list(summary.keys())}")
    if train_mode not in sim:
        raise KeyError(f"train_mode '{train_mode}' not found in similarity modes: {list(sim.keys())}")

    test_modes = [m for m in modes if m in summary[train_key]]
    if not test_modes:
        raise ValueError("No overlapping test modes between similarity.json and gate summary")

    x = np.array([float(sim[train_mode][m]["mean"]) for m in test_modes], dtype=np.float64)
    y = np.array([float(summary[train_key][m]) for m in test_modes], dtype=np.float64)

    r_p = _pearson(x, y)
    r_s = _spearman(x, y)

    out_dir = args.out_dir or os.path.dirname(args.similarity_json)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "similarity_accuracy_corr.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "train_key": train_key,
                "train_mode_for_similarity": train_mode,
                "test_modes": test_modes,
                "similarity": x.tolist(),
                "accuracy": y.tolist(),
                "pearson_r": float(r_p),
                "spearman_r": float(r_s),
            },
            f,
            indent=2,
        )

    print(f"Pearson r:  {r_p:.3f}")
    print(f"Spearman r: {r_s:.3f}")
    print(f"Saved: {out_json}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        out_png = os.path.join(out_dir, "similarity_vs_accuracy.png")
        plt.figure()
        plt.scatter(x, y)
        for i, m in enumerate(test_modes):
            plt.annotate(m, (x[i], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=9)
        plt.xlabel(f"cosine(sim({train_mode}, test))")
        plt.ylabel(f"accuracy(train={train_key}, test)")
        plt.title(f"Similarity vs accuracy (Pearson={r_p:.2f}, Spearman={r_s:.2f})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        print(f"Saved: {out_png}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
