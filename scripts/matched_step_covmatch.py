"""Coverage-matched re-score. EXPLORATORY: not in the pre-registration.

At a fixed 0.5 threshold the three runs call very different amounts of ink, and
IoU cannot exceed the smaller coverage divided by the larger, so a pair whose runs
disagree about how MUCH ink exists cannot score well however well they agree about
WHERE it is. s42_020k appears in both pairs, so both are re-scored with the other
run thresholded to call exactly as much ink as s42_020k does on that surface. That
removes the ceiling difference and leaves only spatial agreement.

This was chosen after seeing the registered result and carries none of the
pre-registration's authority. See prereg/matched-step-result.md.

Usage: python -I matched_step_covmatch.py <maps_root> <out.csv>
"""

import csv
import glob
import os
import sys

import numpy as np
import tifffile

from inkrepro.metrics import align, to_unit

NAMES = ("s42_020k", "s43_020k", "s43_060k")
FIELDS = ["scroll", "mesh", "direction", "pair", "target_cov", "thr_other",
          "frac_ink_other", "iou_matched_cov"]


def main(maps_root, out_csv):
    rows = []
    for scroll_dir in sorted(glob.glob(os.path.join(maps_root, "*"))):
        scroll = os.path.basename(scroll_dir)
        for mesh_dir in sorted(glob.glob(scroll_dir + "/*")):
            mesh = os.path.basename(mesh_dir)
            for direction in ("forward", "reverse"):
                sfx = "_reverse" if direction == "reverse" else ""
                paths = {n: os.path.join(mesh_dir, n + sfx + ".tif") for n in NAMES}
                mpath = os.path.join(mesh_dir, "valid_mask.tif")
                if not all(os.path.exists(p) for p in list(paths.values()) + [mpath]):
                    print("MISSING", scroll, mesh, direction)
                    continue
                imgs = [tifffile.imread(paths[n]) for n in NAMES]
                cropped = align(*imgs, tifffile.imread(mpath))
                m = cropped[-1].astype(bool)
                v = {n: to_unit(c)[m] for n, c in zip(NAMES, cropped[:-1])}
                ref = v["s42_020k"]
                target = float((ref >= 0.5).mean())   # what the reference calls ink
                ink_ref = ref >= 0.5
                for other, pair in (("s43_020k", "matched_step"), ("s43_060k", "mixed")):
                    o = v[other]
                    # Threshold o so it calls exactly the same fraction as the reference.
                    thr = float(np.quantile(o, 1.0 - target)) if 0 < target < 1 else 0.5
                    ink_o = o >= thr
                    inter = int((ink_ref & ink_o).sum())
                    union = int((ink_ref | ink_o).sum())
                    rows.append(dict(scroll=scroll, mesh=mesh, direction=direction, pair=pair,
                                     target_cov=target, thr_other=thr,
                                     frac_ink_other=float(ink_o.mean()),
                                     iou_matched_cov=(1.0 if union == 0 else inter / union)))
                print("  %s/%s %s target_cov %.4f" % (scroll, mesh, direction, target), flush=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("WROTE", len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
