"""Plot the distribution of cross-checkpoint agreement over the atlas.

Usage:
    python scripts/plot_population.py results/population.csv figures/population.png
"""

from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Measured by inkrepro on the curated PHerc0139 control surface with the same
# checkpoint pair. It is not part of the atlas, so it is drawn as a reference
# line rather than included in the histogram.
W043_CONTROL = 0.4093


def _tags(row: dict) -> tuple[str, ...]:
    return tuple(
        sorted(x.removesuffix("_reverse") for x in (row["run_a"], row["run_b"]))
    )


def main(argv: list[str]) -> int:
    rows = [r for r in csv.DictReader(open(argv[1])) if r["comparison"] == "checkpoint"]
    indep = np.array(
        [float(r["iou"]) for r in rows if _tags(r) == ("s42_020k", "s43_060k")]
    )
    soup = np.array(
        [float(r["iou"]) for r in rows if _tags(r) == ("s42_020k", "soup42_early3")]
    )
    surfaces = len({(r["scroll"], r["mesh"]) for r in rows})

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bins = np.linspace(0, 0.5, 60)
    ax.hist(
        indep,
        bins=bins,
        color="#4a6fa5",
        edgecolor="white",
        linewidth=0.4,
        label=f"independent seeds, seed42 vs seed43 (n={len(indep)})",
    )
    ax.hist(
        soup,
        bins=bins,
        color="#b8b8b8",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label=f"positive control: seed42 vs a soup derived from it (n={len(soup)})",
    )

    med = float(np.median(indep))
    ax.axvline(med, color="#1f3a68", linewidth=2.0)
    ax.annotate(
        f"atlas median {med:.3f}",
        xy=(med, ax.get_ylim()[1] * 0.92),
        xytext=(med + 0.010, ax.get_ylim()[1] * 0.97),
        color="#1f3a68",
        fontweight="bold",
        fontsize=10,
    )
    ax.axvline(W043_CONTROL, color="#c0392b", linewidth=2.0, linestyle="--")
    ax.annotate(
        f"curated on-sheet control\nw043 (PHerc0139) {W043_CONTROL:.3f}",
        xy=(W043_CONTROL, ax.get_ylim()[1] * 0.60),
        xytext=(W043_CONTROL - 0.150, ax.get_ylim()[1] * 0.70),
        color="#c0392b",
        fontweight="bold",
        fontsize=10,
    )

    above = int((indep >= W043_CONTROL).sum())
    ax.set_xlabel(
        "IoU between the two ink maps  (of every pixel either run called ink, "
        "the fraction both called ink)",
        fontsize=10,
    )
    ax.set_ylabel("comparisons", fontsize=10)
    ax.set_title(
        f"Two ink checkpoints, {surfaces} surfaces across the eight eligible scrolls:\n"
        f"they agree on about a sixth of the ink they call",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
              fontsize=9.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.005,
        f"ink = p >= 0.5, measured over each surface's valid mask;  "
        f"{above} of {len(indep)} atlas comparisons reach the curated control;  "
        f"5 of 340 surfaces excluded (maps held, and they are exactly the 5 "
        f"ink-screen candidates)",
        ha="center",
        fontsize=8.5,
        color="#444",
    )
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(argv[2], dpi=140, facecolor="white", bbox_inches="tight")
    print(f"wrote {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
