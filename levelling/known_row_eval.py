"""known_row_eval.py <patch.tifxyz> <reference.tifxyz> <letter_boxes.json> <map.tif> [<map.tif> ...] [--csv out.csv]

Score ink maps of a patch against letters whose position is known on ANOTHER surface of the same sheet.

Used here with the PHerc0343 First Letters reading: <reference.tifxyz> is the winner's published mesh and
<letter_boxes.json> his 18 letter boxes ({"letters": [{"id": 1, "box": [x0, y0, x1, y1]}, ...]}, in pixels of his
render; https://github.com/Nieuwlaar/pherc343-first-letters). <patch.tifxyz> is any other surface of that sheet, for
example a patch grown from a seed, turned to any angle. The boxes are carried over in 3D: every point of the
reference mesh inside a box is matched to the nearest point of the patch; matches within 6 voxels give that
letter's pixels in the patch's render (render pixel = grid index / scale). So the score does not depend on how the
patch's grid is turned.

For each map (uint8, 0-255, same shape as the patch's render):
  share of valid pixels (> 8) at >= 128, >= 170 and >= 200 inside the letter boxes, and at >= 128 and >= 170 in the
  band around them (within 400 px of the letters, not on them), plus a rank AUC of box pixels against band pixels.
A patch on another sheet matches few points; that is reported and nothing is scored.
Numbers only. Whether something is a letter is for a person to judge.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import tifffile
from scipy.spatial import cKDTree


def load(path):
    X, Y, Z = (tifffile.imread(os.path.join(path, f"{a}.tif")).astype(np.float64) for a in "xyz")
    return X, Y, Z, (Z > 0) & (X > 0) & (Y > 0)


def cell_points(X, Y, Z, V, k=5, rows=None, cols=None):
    """k x k points per valid grid cell with their fractional (row, col); optional grid-index window"""
    H, W = Z.shape
    r0, r1 = rows if rows else (0, H - 1)
    c0, c1 = cols if cols else (0, W - 1)
    r0, c0 = max(0, int(np.floor(r0))), max(0, int(np.floor(c0)))
    r1, c1 = min(H - 1, int(np.ceil(r1))), min(W - 1, int(np.ceil(c1)))
    q = V[r0:r1, c0:c1] & V[r0 + 1:r1 + 1, c0:c1] & V[r0:r1, c0 + 1:c1 + 1] & V[r0 + 1:r1 + 1, c0 + 1:c1 + 1]
    ii, jj = np.nonzero(q)
    if len(ii) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))
    ii, jj = ii + r0, jj + c0
    f = (np.arange(k) + 0.5) / k
    P, R = [], []
    for fi in f:
        for fj in f:
            P.append(np.stack([A[ii, jj] * (1 - fi) * (1 - fj) + A[ii + 1, jj] * fi * (1 - fj) + A[ii, jj + 1] * (1 - fi) * fj
                               + A[ii + 1, jj + 1] * fi * fj for A in (X, Y, Z)], 1))
            R.append(np.stack([ii + fi, jj + fj], 1))
    return np.concatenate(P), np.concatenate(R)


def rank_auc(pos, neg, rng, n=300000):
    """P(a box pixel is brighter than a band pixel), ties counted half"""
    if len(pos) > n:
        pos = rng.choice(pos, n, replace=False)
    if len(neg) > n:
        neg = rng.choice(neg, n, replace=False)
    v = np.concatenate([pos, neg]).astype(np.float64)
    r = np.empty(len(v))
    r[v.argsort(kind="mergesort")] = np.arange(1, len(v) + 1)
    u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    s = np.zeros(len(u))
    np.add.at(s, inv, r)
    r = (s / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def letter_masks(patch, reference, boxes, reach=6.0):
    """(letters, band, per-letter match share) in the patch's render frame"""
    X, Y, Z, V = load(patch)
    s = float(json.load(open(os.path.join(patch, "meta.json")))["scale"][0])
    P, R = cell_points(X, Y, Z, V)
    tree = cKDTree(P)
    WX, WY, WZ, WV = load(reference)
    ws = float(json.load(open(os.path.join(reference, "meta.json")))["scale"][0])
    H, W = round(X.shape[0] / s), round(X.shape[1] / s)
    mask = np.zeros((H, W), np.uint8)
    per = []
    for L in json.load(open(boxes))["letters"]:
        x0, y0, x1, y1 = L["box"]
        pts, _ = cell_points(WX, WY, WZ, WV, rows=(y0 * ws, y1 * ws), cols=(x0 * ws, x1 * ws))
        if len(pts) == 0:
            per.append((L["id"], 0.0))
            continue
        d, k = tree.query(pts)
        ok = d <= reach
        rc = R[k[ok]] / s
        mask[np.clip(np.round(rc[:, 0]).astype(int), 0, H - 1), np.clip(np.round(rc[:, 1]).astype(int), 0, W - 1)] = 255
        per.append((L["id"], float(ok.mean())))
    mask = cv2.morphologyEx(cv2.dilate(mask, np.ones((9, 9), np.uint8)), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    letters = mask > 0
    band = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (801, 801))) > 0
    band &= ~cv2.dilate(mask, np.ones((61, 61), np.uint8)).astype(bool)
    return letters, band, per


def score(m, letters, band, rng):
    valid = m > 8
    a, b = m[letters & valid], m[band & valid]
    if len(a) < 1000 or len(b) < 1000:
        return None
    return dict(auc_boxes_vs_band=rank_auc(a, b, rng), share_ge128_in_boxes=float((a >= 128).mean()), share_ge128_band=float((b >= 128).mean()),
                share_ge170_in_boxes=float((a >= 170).mean()), share_ge170_band=float((b >= 170).mean()), share_ge200_in_boxes=float((a >= 200).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score ink maps of a patch against letters known on another surface of the same sheet.")
    ap.add_argument("patch")
    ap.add_argument("reference")
    ap.add_argument("boxes")
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    letters, band, per = letter_masks(a.patch, a.reference, a.boxes)
    print("letters matched (share of each box's points within 6 voxels of the patch): " + " ".join(f"{i}:{f:.2f}" for i, f in per))
    print(f"letter pixels {int(letters.sum())}, band pixels {int(band.sum())}")
    if letters.sum() < 20000:
        print("too few letter pixels on this patch: it is not on the text sheet there")
        return
    rng = np.random.default_rng(0)
    rows = []
    for f in a.maps:
        m = tifffile.imread(f)
        m = m[0] if m.ndim == 3 else m
        if m.shape != letters.shape:
            print("shape mismatch", os.path.basename(f), m.shape, letters.shape)
            continue
        r = score(m, letters, band, rng)
        if r is None:
            print("too few valid pixels", os.path.basename(f))
            continue
        rows.append(dict(map=os.path.basename(f), **r))
        print(f"  >=128 in/band {r['share_ge128_in_boxes']:.3f}/{r['share_ge128_band']:.3f}  >=170 in/band {r['share_ge170_in_boxes']:.3f}/{r['share_ge170_band']:.3f}  "
              f">=200 in {r['share_ge200_in_boxes']:.4f}  auc {r['auc_boxes_vs_band']:.3f}  {os.path.basename(f)}")
    if a.csv and rows:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
