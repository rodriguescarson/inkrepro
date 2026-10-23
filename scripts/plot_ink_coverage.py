"""Agreement against how much ink a surface calls.

Cross-checkpoint IoU is not a free-standing confidence number: it rises steeply
with the fraction of the surface either run calls ink. Any comparison between
two surfaces has to match on coverage first, which is what this plot is for.

Usage:
    python scripts/plot_ink_coverage.py results/population.csv \
        figures/ink_coverage.png [results/held_out_control.csv]
"""

from __future__ import annotations

import csv
import math
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path, forward_only=True):
    out = []
    for r in csv.DictReader(open(path)):
        if r["comparison"] != "checkpoint":
            continue
        if tuple(sorted(x.removesuffix("_reverse") for x in (r["run_a"], r["run_b"]))) \
           != ("s42_020k", "s43_060k"):
            continue
        if forward_only and r["run_a"].endswith("_reverse"):
            continue
        out.append(
            ((float(r["frac_ink_a"]) + float(r["frac_ink_b"])) / 2,
             float(r["iou"]), r["mesh"])
        )
    return out


def _spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(s):
            out[i] = pos
        return out
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def main(argv):
    atlas = _rows(argv[1])
    xs = [a for a, _, _ in atlas]
    ys = [b for _, b, _ in atlas]
    rho = _spearman(xs, ys)
    held = _rows(argv[3]) if len(argv) > 3 else []

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.scatter(xs, ys, s=16, color="#4a6fa5", alpha=0.45, edgecolor="none",
               label=f"atlas surfaces, forward (n={len(atlas)})")

    # median IoU in equal-count coverage bins: the trend without a fitted model
    order = sorted(zip(xs, ys))
    nbins = 10
    edges = [round(i * len(order) / nbins) for i in range(nbins + 1)]
    bx, by = [], []
    for lo, hi in zip(edges, edges[1:]):
        chunk = order[lo:hi]
        if not chunk:
            continue
        bx.append(st.median([a for a, _ in chunk]))
        by.append(st.median([b for _, b in chunk]))
    ax.plot(bx, by, color="#1f3a68", linewidth=2.2, marker="o", markersize=4,
            label="atlas median in equal-count coverage bins")

    # label above or below alternately, so neighbouring windings do not collide
    for i, (cov, iou, mesh) in enumerate(sorted(held)):
        ax.scatter([cov], [iou], s=90, marker="D", color="#c0392b",
                   edgecolor="white", linewidth=1.0, zorder=5)
        dy = 0.009 if i % 2 == 0 else -0.015
        ax.annotate(mesh, xy=(cov, iou), xytext=(cov + 0.005, iou + dy),
                    color="#c0392b", fontweight="bold", fontsize=9, zorder=6)
    if held:
        ax.scatter([], [], s=90, marker="D", color="#c0392b", edgecolor="white",
                   label=f"held-out on-sheet PHerc0139 windings (n={len(held)})")

    ax.set_xlabel("ink coverage — mean fraction of the valid region each run called ink",
                  fontsize=10)
    ax.set_ylabel("IoU between the two checkpoints", fontsize=10)
    ax.set_title(
        "Agreement is mostly a function of how much ink is called\n"
        f"Spearman rho = {rho:+.2f} over {len(atlas)} atlas surfaces",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.text(0.5, 0.005,
             "A surface that calls more ink scores higher almost mechanically: when ink is "
             "sparse the union is dominated by stroke-edge disagreement.\n"
             "Comparing two surfaces therefore means comparing them at matched coverage, "
             "not comparing their raw IoU.",
             ha="center", fontsize=8.5, color="#444")
    fig.subplots_adjust(bottom=0.17)
    fig.savefig(argv[2], dpi=140, facecolor="white", bbox_inches="tight")
    print(f"wrote {argv[2]}  (Spearman {rho:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
