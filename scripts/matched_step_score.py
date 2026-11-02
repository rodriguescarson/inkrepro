"""Score one surface's three runs against each other. Ran on the pod; see matched_step_run.sh.

Uses this repository's own inkrepro.metrics so the batch cannot drift from the
published numbers. Validated before use: on PHerc0125 z14144_w040 it reproduces
the sample's recorded mixed-pair IoU 0.13649655597278823 exactly.

Usage: python -I matched_step_score.py <maps_dir> <scroll> <mesh> <out.csv>
"""

import csv
import os
import sys

import tifffile

from inkrepro.metrics import agreement, align, to_unit

NAMES = ["s42_020k", "s43_020k", "s43_060k"]
FIELDS = ["scroll", "mesh", "pair", "direction", "run_a", "run_b", "n_valid", "iou",
          "dice", "pearson_r", "frac_ink_a", "frac_ink_b", "mean_abs_diff"]


def main(work, scroll, mesh, out_csv):
    rows = []
    for direction in ("forward", "reverse"):
        suffix = "_reverse" if direction == "reverse" else ""
        paths = {n: os.path.join(work, n + suffix + ".tif") for n in NAMES}
        if not all(os.path.exists(p) for p in paths.values()):
            missing = [n for n, p in paths.items() if not os.path.exists(p)]
            print("MISSING", scroll, mesh, direction, missing)
            continue
        imgs = [tifffile.imread(paths[n]) for n in NAMES]
        mask = tifffile.imread(os.path.join(work, "valid_mask.tif"))
        # One crop for everything: inference pads the canvas to a patch multiple,
        # so a map comes back larger than the mask written beside it.
        cropped = align(*imgs, mask)
        m = cropped[-1].astype(bool)
        unit = {n: to_unit(c) for n, c in zip(NAMES, cropped[:-1])}
        for a, b, pair in (("s42_020k", "s43_020k", "matched_step"),
                           ("s42_020k", "s43_060k", "mixed")):
            g = agreement(unit[a], unit[b], mask=m)
            rows.append(dict(scroll=scroll, mesh=mesh, pair=pair, direction=direction,
                             run_a=a, run_b=b, n_valid=g.n_valid, iou=g.iou, dice=g.dice,
                             pearson_r=g.pearson_r, frac_ink_a=g.frac_ink_a,
                             frac_ink_b=g.frac_ink_b, mean_abs_diff=g.mean_abs_diff))
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    for r in rows:
        print("  %-12s %-7s IoU %.6f  n_valid %d" % (r["pair"], r["direction"], r["iou"], r["n_valid"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
