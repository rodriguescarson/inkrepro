"""Compare each held-out winding against atlas surfaces that call a similar amount of ink.

Agreement rises with ink coverage (Spearman +0.83 over the 335 atlas surfaces), so a
winding beating the population median proves nothing on its own. Each winding is
compared only with atlas surfaces whose coverage -- the mean of the two runs' ink
fractions -- is within a factor of 1.25 either way of its own.

Usage: python scripts/ink_matched.py results/held_out_control.csv results/population.csv

Reverse reads the opposite face, where there should be less real ink to carry the
agreement. If the held-out windings' advantage is ink rather than an artefact of the
surfaces, it should shrink here. Held reverse rows are matched against the atlas's own
reverse rows, so both sides read the same face.
"""
import csv, statistics, sys

def rows(path, reverse):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["comparison"] != "checkpoint" or "soup" in r["run_a"] + r["run_b"]:
                continue
            if r["run_a"].endswith("_reverse") != reverse:
                continue
            r["coverage"] = (float(r["frac_ink_a"]) + float(r["frac_ink_b"])) / 2
            r["iou_f"] = float(r["iou"])
            yield r

def table(held_path, pop_path, reverse, label):
    pop = list(rows(pop_path, reverse))
    held = list(rows(held_path, reverse))
    print(f"\n{label}  (atlas n={len(pop)})")
    print(f"{'winding':<8} {'coverage':>9} {'IoU':>7} {'band median':>12} {'n':>5} {'delta':>8} {'pct':>7}")
    ds, ps = [], []
    for h in held:
        lo, hi = h["coverage"] / 1.25, h["coverage"] * 1.25
        band = [p["iou_f"] for p in pop if lo <= p["coverage"] <= hi]
        med = statistics.median(band)
        pct = 100.0 * sum(1 for v in band if v < h["iou_f"]) / len(band)
        ds.append(h["iou_f"] - med); ps.append(pct)
        print(f"{h['mesh']:<8} {h['coverage']:>9.3f} {h['iou_f']:>7.3f} {med:>12.3f} "
              f"{len(band):>5} {h['iou_f'] - med:>+8.3f} {pct:>6.1f}%")
    print(f"  range: {min(ds):+.3f} to {max(ds):+.3f}   percentiles {min(ps):.0f}-{max(ps):.0f}")

if __name__ == "__main__":
    table(sys.argv[1], sys.argv[2], False, "FORWARD")
    table(sys.argv[1], sys.argv[2], True, "REVERSE")
