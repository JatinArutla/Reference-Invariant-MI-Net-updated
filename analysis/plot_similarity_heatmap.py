#!/usr/bin/env python
"""Plot embedding similarity heatmap produced by analysis/embedding_drift.py."""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser("Plot similarity heatmap")
    ap.add_argument("--similarity_json", type=str, required=True)
    ap.add_argument("--out_png", type=str, default="")
    ap.add_argument("--title", type=str, default="Embedding cosine similarity")
    args = ap.parse_args()

    with open(args.similarity_json, "r") as f:
        obj = json.load(f)

    modes = obj.get("ref_modes") or list((obj.get("pairwise_cosine_similarity") or {}).keys())
    sim = obj["pairwise_cosine_similarity"]

    M = [[float(sim[a][b]["mean"]) for b in modes] for a in modes]

    out_png = args.out_png or os.path.join(os.path.dirname(args.similarity_json), "similarity_heatmap.png")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required to plot") from e

    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(modes)), modes, rotation=45, ha="right")
    plt.yticks(range(len(modes)), modes)
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
