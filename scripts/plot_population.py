"""Plot the distribution of cross-checkpoint agreement over the atlas.

Usage:
    python scripts/plot_population.py results/population.csv figures/population.png \
        [results/held_out_control.csv]

The held-out control file is optional; when given, each held-out PHerc0139
winding is drawn as its own marker rather than as a single summary line, because
four surfaces are four data points and should look like four data points.
"""

from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _tags(row: dict) -> tuple[str, ...]:
    return tuple(
        sorted(x.removesuffix("_reverse") for x in (row["run_a"], row["run_b"]))
    )


def _forward(row: dict) -> bool:
    return not row["run_a"].endswith("_reverse")


def _read(path: str, pair: tuple[str, str], forward_only: bool) -> list[dict]:
    return [
        r
        for r in csv.DictReader(open(path))
        if r["comparison"] == "checkpoint"
        and _tags(r) == pair
        and (not forward_only or _forward(r))
    ]


def main(argv: list[str]) -> int:
    rows = _read(argv[1], ("s42_020k", "s43_060k"), False)
    indep = np.array([float(r["iou"]) for r in rows])
    soup = np.array(
        [float(r["iou"]) for r in _read(argv[1], ("s42_020k", "soup42_early3"), False)]
    )
    surfaces = len({(r["scroll"], r["mesh"]) for r in rows})

    held = []
    if len(argv) > 3:
        held = [
            (r["mesh"], float(r["iou"]))
            for r in _read(argv[3], ("s42_020k", "s43_060k"), True)
        ]
        held.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    bins = np.linspace(0, 0.5, 60)
    ax.hist(
        indep,
        bins=bins,
        color="#4a6fa5",
        edgecolor="white",
        linewidth=0.4,
        label=f"atlas, independent seeds seed42 vs seed43 (n={len(indep)})",
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
    top = ax.get_ylim()[1]
    ax.axvline(med, color="#1f3a68", linewidth=2.0)
    ax.annotate(
        f"atlas median {med:.3f}",
        xy=(med, top * 0.92),
        xytext=(med + 0.010, top * 0.97),
        color="#1f3a68",
        fontweight="bold",
        fontsize=10,
    )

    for i, (mesh, value) in enumerate(held):
        ax.axvline(value, color="#c0392b", linewidth=1.6, linestyle="--", alpha=0.9)
        ax.annotate(
            mesh,
            xy=(value, top * (0.42 - 0.07 * i)),
            xytext=(value + 0.006, top * (0.42 - 0.07 * i)),
            color="#c0392b",
            fontweight="bold",
            fontsize=9,
        )
    if held:
        ax.plot(
            [], [], color="#c0392b", linewidth=1.6, linestyle="--",
            label=f"held-out on-sheet PHerc0139 windings, forward (n={len(held)})",
        )

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
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
        fontsize=9.5, frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.005,
        "ink = p >= 0.5, measured over each surface's valid mask. 5 of 340 surfaces are "
        "excluded: their maps are held back, and they are exactly the 5 ink-screen candidates.\n"
        "IoU rises steeply with how much ink a surface calls (Spearman +0.83), so raw values "
        "are not comparable across surfaces. The four held-out windings are\nmarked here for "
        "position only; the comparison that controls for coverage is in figures/ink_coverage.png.",
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
