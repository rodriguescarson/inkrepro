"""Regenerate results/SUMMARY.txt from the committed CSVs.

Usage:
    python scripts/write_summary.py results/population.csv \
        results/held_out_control.csv results/SUMMARY.txt
"""

from __future__ import annotations

import csv
import math
import statistics as st
import sys


def tags(r):
    return tuple(sorted(x.removesuffix("_reverse") for x in (r["run_a"], r["run_b"])))


def cov(r):
    return (float(r["frac_ink_a"]) + float(r["frac_ink_b"])) / 2


def pair(rows, a, b, forward=None):
    out = [r for r in rows if r["comparison"] == "checkpoint" and tags(r) == (a, b)]
    if forward is not None:
        out = [r for r in out if r["run_a"].endswith("_reverse") != forward]
    return out


def spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        o = [0] * len(v)
        for p, i in enumerate(s):
            o[i] = p
        return o
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def main(pop_path, held_path, out_path):
    rows = list(csv.DictReader(open(pop_path)))
    held = list(csv.DictReader(open(held_path)))
    ind = pair(rows, "s42_020k", "s43_060k")
    soup = pair(rows, "s42_020k", "soup42_early3")
    iou = [float(r["iou"]) for r in ind]
    q = st.quantiles(iou, n=4, method="inclusive")
    fwd = [float(r["iou"]) for r in pair(rows, "s42_020k", "s43_060k", forward=True)]
    rev = [float(r["iou"]) for r in pair(rows, "s42_020k", "s43_060k", forward=False)]
    L = []
    w = L.append
    w("inkrepro - cross-checkpoint agreement over the eligible-scroll atlas")
    w("")
    w(f"surfaces measured          {len({(r['scroll'], r['mesh']) for r in ind})} of 340")
    w("excluded                     5 (maps_held=True; exactly the 5 ink_pass_v1 candidates)")
    w(f"comparisons (indep seeds)  {len(ind)}  (forward and reverse)")
    w("")
    w("INDEPENDENT SEEDS  s42_020k vs s43_060k")
    w(f"  median IoU   {st.median(iou):.4f}")
    w(f"  IQR          {q[0]:.4f} - {q[2]:.4f}")
    w(f"  range        {min(iou):.4f} - {max(iou):.4f}")
    w(f"  forward      {st.median(fwd):.4f}      reverse {st.median(rev):.4f}")
    w("")
    si = [float(r["iou"]) for r in soup]
    w("POSITIVE CONTROL  s42_020k vs soup42_early3 (soup derived from seed42)")
    w(f"  median IoU   {st.median(si):.4f}   ratio to independent "
      f"{st.median(si)/st.median(iou):.2f}x")
    w("")
    f_ind = pair(rows, "s42_020k", "s43_060k", forward=True)
    xs, ys = [cov(r) for r in f_ind], [float(r["iou"]) for r in f_ind]
    w("CONFOUND  agreement tracks how much ink is called")
    w(f"  Spearman rho(coverage, IoU) = {spearman(xs, ys):+.3f}   (forward, n={len(xs)})")
    order = sorted(zip(xs, ys))
    k = len(order) // 4
    for i, lab in enumerate(("Q1 least ink", "Q2          ", "Q3          ", "Q4 most ink ")):
        c = order[i * k:(i + 1) * k] if i < 3 else order[3 * k:]
        w(f"  {lab}  coverage {st.median([a for a, _ in c]):.4f}  "
          f"median IoU {st.median([b for _, b in c]):.4f}")
    w("")
    w("HELD-OUT ON-SHEET CONTROL  PHerc0139 windings in neither ink_9um training list")
    w("  (masks reconstructed; in-band reconstruction error mean -0.010, max 0.047)")
    for r in sorted(pair(held, "s42_020k", "s43_060k", forward=True),
                    key=lambda r: r["mesh"]):
        band = [float(x["iou"]) for x in f_ind if 0.8 * cov(r) <= cov(x) <= 1.25 * cov(r)]
        w(f"  {r['mesh']}  coverage {cov(r):.4f}  IoU {float(r['iou']):.4f}  "
          f"ink-matched atlas median {st.median(band):.4f} (n={len(band)})  "
          f"delta {float(r['iou'])-st.median(band):+.4f}")
    w("")
    w("per scroll (independent seeds, forward and reverse):")
    w(f"  {'scroll':<12} {'n':>4} {'median':>8} {'min':>8} {'max':>8}")
    by = {}
    for r in ind:
        by.setdefault(r["scroll"], []).append(float(r["iou"]))
    for s in sorted(by):
        v = by[s]
        w(f"  {s:<12} {len(v):>4} {st.median(v):>8.4f} {min(v):>8.4f} {max(v):>8.4f}")
    open(out_path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
