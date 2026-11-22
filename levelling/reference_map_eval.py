"""reference_map_eval.py <reference_mesh.tifxyz> <reference_probability.tif> <meshes_dir> <maps_dir>

Score ink maps of OTHER renders of a sheet against a reference probability map made on one surface of that sheet.

Used here with the PHerc0211 First Letters reading: <reference_mesh.tifxyz> is the winner's three-row surface and
<reference_probability.tif> his published probability map on it (float32, 0-1). <meshes_dir> holds the same surface
turned 0 / 15 / 30 / 45 degrees with level_tifxyz.py --angle, one tifxyz folder per tag; <maps_dir> holds our maps of
the renders of those folders, named <tag>__<model>__d+0[_reverse].tif (uint8).
For every 4th pixel of a turned render, its 3D point (bilinear on the turned grid) is matched to the nearest point of
the reference mesh (within 6 voxels) and the reference probability is read there, so the score does not depend on
how the mesh is turned. ink = reference >= 0.5, blank = reference <= 0.1. Per map: Pearson r with the reference, the
share of our pixels >= 128 / >= 170 on reference ink and on reference blank, the share >= 200 on reference ink, and
the rank AUC of ink against blank.
Numbers only. Whether something is a letter is for a person to judge.
"""
import glob, json, os, sys
import numpy as np, tifffile
from scipy.spatial import cKDTree
def load(d):
    X, Y, Z = (tifffile.imread(os.path.join(d, f"{a}.tif")).astype(np.float64) for a in "xyz")
    return X, Y, Z, (Z > 0) & (X > 0) & (Y > 0)
def cells(X, Y, Z, V, k=5):
    q = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]; ii, jj = np.nonzero(q); f = (np.arange(k) + 0.5) / k; P, R = [], []
    for fi in f:
        for fj in f:
            P.append(np.stack([A[ii, jj] * (1 - fi) * (1 - fj) + A[ii + 1, jj] * fi * (1 - fj) + A[ii, jj + 1] * (1 - fi) * fj + A[ii + 1, jj + 1] * fi * fj for A in (X, Y, Z)], 1))
            R.append(np.stack([ii + fi, jj + fj], 1))
    return np.concatenate(P), np.concatenate(R)
def auc(pos, neg, rng, n=200000):
    pos = rng.choice(pos, min(n, len(pos)), replace=False); neg = rng.choice(neg, min(n, len(neg)), replace=False)
    v = np.concatenate([pos, neg]).astype(np.float64); r = np.empty(len(v)); r[v.argsort(kind="mergesort")] = np.arange(1, len(v) + 1)
    u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True); s = np.zeros(len(u)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
hm, hp, md, mapd = sys.argv[1:5]
HX, HY, HZ, HV = load(hm); hs = float(json.load(open(os.path.join(hm, "meta.json")))["scale"][0])
HP, HR = cells(HX, HY, HZ, HV); tree = cKDTree(HP)
H = tifffile.imread(hp).astype(np.float32); H = H[0] if H.ndim == 3 else H
print(f"REFERENCE mesh grid {HX.shape}, map {H.shape}, ink {np.mean(H >= 0.5):.4f}, blank {np.mean(H <= 0.1):.4f}", flush=True)
rng = np.random.default_rng(0); STEP = 4
for tag in sorted(os.listdir(md)):
    d = os.path.join(md, tag)
    if not os.path.exists(os.path.join(d, "x.tif")): continue
    X, Y, Z, V = load(d); s = hs
    rows, cols = round(X.shape[0] / s), round(X.shape[1] / s)
    yy, xx = np.mgrid[0:rows:STEP, 0:cols:STEP]
    r, c = yy.ravel() * s, xx.ravel() * s
    r0, c0 = np.clip(np.floor(r).astype(int), 0, X.shape[0] - 2), np.clip(np.floor(c).astype(int), 0, X.shape[1] - 2)
    fr, fc = r - r0, c - c0
    ok = V[r0, c0] & V[r0 + 1, c0] & V[r0, c0 + 1] & V[r0 + 1, c0 + 1]
    P = np.stack([A[r0, c0] * (1 - fr) * (1 - fc) + A[r0 + 1, c0] * fr * (1 - fc) + A[r0, c0 + 1] * (1 - fr) * fc + A[r0 + 1, c0 + 1] * fr * fc for A in (X, Y, Z)], 1)
    dist, k = tree.query(P[ok], distance_upper_bound=6.0, workers=-1)
    hit = np.isfinite(dist)
    hy = np.clip(np.round(HR[k[hit], 0] / hs).astype(int), 0, H.shape[0] - 1); hx = np.clip(np.round(HR[k[hit], 1] / hs).astype(int), 0, H.shape[1] - 1)
    ref = H[hy, hx]
    py, px = yy.ravel()[ok][hit], xx.ravel()[ok][hit]
    ink, blank = ref >= 0.5, ref <= 0.1
    print(f"MESH {tag}: grid {X.shape}, render {rows}x{cols}, sampled {int(ok.sum())} px, matched within 6 voxels {hit.mean():.3f}, his ink share there {ink.mean():.4f}", flush=True)
    for f in sorted(glob.glob(os.path.join(mapd, tag + "__*.tif"))):
        M = tifffile.imread(f); M = M[0] if M.ndim == 3 else M
        if abs(M.shape[0] - rows) > 2 or abs(M.shape[1] - cols) > 2:
            print(f"   {os.path.basename(f)} shape {M.shape} does not fit the render {rows}x{cols}"); continue
        A = M[np.clip(py, 0, M.shape[0] - 1), np.clip(px, 0, M.shape[1] - 1)].astype(np.float32); v = A > 8
        i, b = ink & v, blank & v
        print(f"   {os.path.basename(f):<40} r {np.corrcoef(A[v], ref[v])[0, 1]:+.3f}  >=128 ink/blank {np.mean(A[i] >= 128):.3f}/{np.mean(A[b] >= 128):.3f}  "
              f">=170 {np.mean(A[i] >= 170):.3f}/{np.mean(A[b] >= 170):.3f}  >=200 ink {np.mean(A[i] >= 200):.4f}  auc {auc(A[i], A[b], rng):.3f}", flush=True)
