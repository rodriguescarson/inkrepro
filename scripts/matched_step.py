"""The pre-registered matched-step readout, and why the registered number is not the answer.

prereg/matched-step.md asked one question: of the disagreement between
`s42_020k` and `s43_060k`, how much is the training seed and how much is the
training step? It froze 40 surfaces, fixed the thresholds of interpretation
before any inference ran, and committed to publishing whichever way it landed.

It landed in the one direction the pre-registration called a surprise: at the
registered fixed 0.5 threshold the matched-step pair agrees *less* than the
mixed pair. This script reports that, and then reports the reason, which is a
property of the measure rather than of the checkpoints.

IoU between two binarised maps is bounded above by the ratio of the two
coverages. `s43_020k` calls about half as much ink as `s42_020k`; `s43_060k`
calls about 1.7x as much. The matched-step pair therefore starts with a lower
ceiling than the mixed pair, by roughly the size of the effect. Re-scoring with
each run thresholded to call exactly as much ink as `s42_020k` removes that
ceiling and leaves only whether the runs agree about *where* the ink is.

The coverage-matched analysis is exploratory. It is not a substitute for the
registered one and neither is reported without the other.

Usage: python -I scripts/matched_step.py results/matched_step.csv \
           results/matched_step_coverage_matched.csv prereg/matched_step_sample.csv
"""

import csv
import math
import random
import sys
from collections import defaultdict


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def sign_test(diffs):
    """Exact two-sided binomial test on the signs. Ties are dropped, as is usual."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def median_ci(values, iters=20000, seed=20261102):
    """Percentile bootstrap CI for the median. Seeded, so the interval is reproducible."""
    rng = random.Random(seed)
    n = len(values)
    meds = sorted(median([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def readout(median_diff):
    """The interpretation fixed in prereg/matched-step.md before the numbers existed."""
    if median_diff < 0:
        return "negative -- the pre-registered surprise case"
    if median_diff <= 0.010:
        return "<= +0.010 -- the step contributes little; the headline is a seed effect"
    if median_diff >= 0.050:
        return ">= +0.050 -- the step carries a large part of the disagreement"
    return "between +0.010 and +0.050 -- partial"


def paired(rows, iou_field):
    """Group rows into one matched-step/mixed pair per surface and direction."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["scroll"], r["mesh"], r["direction"])][r["pair"]] = r
    out = defaultdict(list)
    for (scroll, mesh, direction), cell in sorted(by.items()):
        if "matched_step" not in cell or "mixed" not in cell:
            continue
        a = float(cell["matched_step"][iou_field])
        b = float(cell["mixed"][iou_field])
        out[direction].append((scroll, mesh, a, b))
    return out


def report(title, groups, coverage=None, registered=False):
    for direction in ("forward", "reverse"):
        entries = groups.get(direction, [])
        if not entries:
            continue
        matched = [a for _, _, a, _ in entries]
        mixed = [b for _, _, _, b in entries]
        diffs = [a - b for _, _, a, b in entries]
        pos, neg, p = sign_test(diffs)
        lo, hi = median_ci(diffs)
        role = "PRIMARY" if direction == "forward" else "secondary"
        print(f"=== {title}: {direction} ({role}), n={len(diffs)} ===")
        print(f"  median matched-step IoU  : {median(matched):.4f}")
        print(f"  median mixed-pair IoU    : {median(mixed):.4f}")
        print(f"  median paired difference : {median(diffs):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  sign test                : {pos} positive, {neg} negative, p = {p:.3g}")
        if registered:
            print(f"  pre-registered readout   : {readout(median(diffs))}")
        else:
            # The registered thresholds were fixed for the registered measure.
            # Applying them to an exploratory re-scoring would launder an
            # after-the-fact analysis into a pre-registered one.
            straddles = lo <= 0 <= hi
            print(f"  (exploratory: no registered threshold applies; "
                  f"CI {'includes' if straddles else 'excludes'} zero)")
        if coverage is not None:
            covs = [coverage[(s, m)] for s, m, _, _ in entries]
            print(f"  Spearman rho, coverage vs matched-step IoU: {spearman(covs, matched):+.3f}")
            print(f"  Spearman rho, coverage vs mixed-pair IoU  : {spearman(covs, mixed):+.3f}")
        print()


def main(registered_path, covmatched_path, sample_path):
    registered = list(csv.DictReader(open(registered_path)))
    covmatched = list(csv.DictReader(open(covmatched_path)))
    coverage = {(r["scroll"], r["mesh"]): float(r["coverage"])
                for r in csv.DictReader(open(sample_path))}

    report("registered, fixed 0.5 threshold", paired(registered, "iou"), coverage,
           registered=True)

    print("=== why: what each run calls ink, at 0.5 ===")
    by = defaultdict(dict)
    for r in registered:
        by[(r["scroll"], r["mesh"], r["direction"])][r["pair"]] = r
    for direction in ("forward", "reverse"):
        ref, new, old = [], [], []
        for (_, _, d), cell in by.items():
            if d != direction or len(cell) < 2:
                continue
            ref.append(float(cell["matched_step"]["frac_ink_a"]))
            new.append(float(cell["matched_step"]["frac_ink_b"]))
            old.append(float(cell["mixed"]["frac_ink_b"]))
        print(f"  {direction}: median frac_ink  s42_020k {median(ref):.4f}  "
              f"s43_020k {median(new):.4f}  s43_060k {median(old):.4f}")
        # Ceiling per surface, then the median of those: IoU cannot exceed the
        # smaller coverage over the larger, whichever run is the larger one.
        ceil_m = median([min(a, b) / max(a, b) for a, b in zip(ref, new) if max(a, b) > 0])
        ceil_x = median([min(a, b) / max(a, b) for a, b in zip(ref, old) if max(a, b) > 0])
        print(f"            median per-surface IoU ceiling  "
              f"matched-step {ceil_m:.3f}  mixed {ceil_x:.3f}")
    print()

    report("exploratory, coverage-matched", paired(covmatched, "iou_matched_cov"))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
