"""Re-measure the four held-out windings under their exact atlas masks.

results/held_out_control.csv was computed with the mask reconstructed from the ink
maps' own footprint, because these four ship no valid_mask.tif. This recomputes the
same 15 pairs per winding against erode(max_z(volume)>0, 64) taken from the published
surface volume, so the held-out result stops carrying a +/-0.047 reconstruction bound.

Usage: python -I recompute_held.py <masks_dir> <maps_dir> <out.csv>
"""

import csv
import itertools
import sys

import numpy as np
import tifffile

from inkrepro.metrics import agreement, align, to_unit

RUNS = {
    "s42_020k": "hybrid_3d2d-seed42_step-020000",
    "s42_020k_reverse": "hybrid_3d2d-seed42_step-020000_reverse",
    "s43_060k": "hybrid_3d2d-seed43_step-060000",
    "s43_060k_reverse": "hybrid_3d2d-seed43_step-060000_reverse",
    "soup42_early3": "soup42_early3",
    "soup42_early3_reverse": "soup42_early3_reverse",
}
WINDINGS = ("w051", "w055", "w058", "w059")
FIELDS = ["scroll", "mesh", "run_a", "run_b", "comparison", "n_valid", "pearson_r",
          "iou", "dice", "frac_ink_a", "frac_ink_b", "mean_abs_diff"]


def comparison(a, b):
    """Same label scheme as the atlas run: what differs between two runs."""
    ckpt_a, rev_a = a.replace("_reverse", ""), a.endswith("_reverse")
    ckpt_b, rev_b = b.replace("_reverse", ""), b.endswith("_reverse")
    if ckpt_a == ckpt_b:
        return "direction"
    return "checkpoint" if rev_a == rev_b else "checkpoint+direction"


def main(masks_dir, maps_dir, out_path):
    rows = []
    for w in WINDINGS:
        z = np.load(f"{masks_dir}/{w}_mask.npz")
        h, wid = (int(x) for x in z["shape"])
        mask = np.unpackbits(z["packed"])[: h * wid].reshape(h, wid).astype(bool)

        maps = {}
        for name, stem in RUNS.items():
            img = tifffile.imread(f"{maps_dir}/0139_{w}_{stem}.tif")
            maps[name] = img
        # One crop for everything: inference pads the canvas to a patch multiple,
        # so a map can come back a few pixels larger than the volume's own grid.
        names = list(maps)
        cropped = align(*[maps[n] for n in names], mask)
        mask = cropped[-1].astype(bool)
        unit = {n: to_unit(c).astype(np.float32) for n, c in zip(names, cropped[:-1])}
        del maps, cropped

        print(f"{w}: canvas {mask.shape}  mask {int(mask.sum()):,} px "
              f"({100 * mask.mean():.1f}% of canvas)", flush=True)

        for a, b in itertools.combinations(names, 2):
            r = agreement(unit[a], unit[b], mask=mask)
            rows.append({"scroll": "PHerc0139", "mesh": w, "run_a": a, "run_b": b,
                         "comparison": comparison(a, b), **r.as_dict()})
            del r
        ck = next(x for x in rows[-15:]
                  if x["run_a"] == "s42_020k" and x["run_b"] == "s43_060k")
        print(f"   checkpoint IoU {ck['iou']:.4f}   ink {ck['frac_ink_a']:.4f}", flush=True)
        del unit

    # The stored column order has threshold dropped; keep the published schema.
    with open(out_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
