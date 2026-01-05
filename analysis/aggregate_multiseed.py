#!/usr/bin/env python
"""Aggregate multiple seeds for the gate benchmark.

Looks for:
  <results_base>/seed_<s>/gate_reference_summary.json

Prints mean and std across seeds for each train\test cell.

Usage:
  python analysis/aggregate_multiseed.py --results_base ./results/jitter_multiseed
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_base", type=str, required=True)
    p.add_argument("--pattern", type=str, default="seed_*", help="Seed directory pattern.")
    args = p.parse_args()

    seed_dirs = sorted([d for d in glob(os.path.join(args.results_base, args.pattern)) if os.path.isdir(d)])
    if not seed_dirs:
        raise SystemExit(f"No seed dirs found under {args.results_base!r}")

    summaries = []
    for d in seed_dirs:
        path = os.path.join(d, "gate_reference_summary.json")
        if not os.path.exists(path):
            print(f"WARN: missing {path}")
            continue
        with open(path, "r") as f:
            obj = json.load(f)
        summaries.append(obj["summary"])

    if not summaries:
        raise SystemExit("No summaries loaded.")

    train_conds = list(summaries[0].keys())
    test_modes = list(next(iter(summaries[0].values())).keys())

    # stack: [S, train, test]
    arr = np.array([
        [[float(s[tr][te]) for te in test_modes] for tr in train_conds]
        for s in summaries
    ], dtype=np.float64)

    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)

    header = "train\\test".ljust(14) + "".join([f"{m:>14}" for m in test_modes])
    print("Mean accuracy (fraction) across seeds")
    print(header)
    print("-" * len(header))
    for i, tr in enumerate(train_conds):
        row = tr.ljust(14) + "".join([f"{mu[i,j]:14.4f}" for j in range(len(test_modes))])
        print(row)

    print("\nStd accuracy (fraction) across seeds")
    print(header)
    print("-" * len(header))
    for i, tr in enumerate(train_conds):
        row = tr.ljust(14) + "".join([f"{sd[i,j]:14.4f}" for j in range(len(test_modes))])
        print(row)


if __name__ == "__main__":
    main()
