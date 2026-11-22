"""level_tifxyz.py <patch.tifxyz> <out.tifxyz> [--angle DEG] [--flip-z] [--center X Y | --umbilicus FILE.json]

Turn a tifxyz surface so that text rows come out level in every render made from it.

Why: a patch from vc_grow_seg_from_seed has grid axes with no relation to the scroll. Text lines run around the
scroll, at constant height z, so on such a patch they run diagonally. Ink models read diagonal text badly: on the
PHerc0343 known letters, reader-v3's crisp stroke pixels halve by about 20 degrees off level and fall 21x by 45
degrees, and three models call ink on the blank side of the sheet (see README.md next to this file).
This rewrites the grid, not the render: the levelled patch goes through vc_render_tifxyz and any ink model unchanged.

How: a plane z ~ a*row + b*col + c is fitted over the valid nodes; its gradient is the direction in which the scroll
axis runs across the grid. x, y and z are resampled on a grid rotated by that angle (bilinear, canvas grown to hold
the whole patch, node spacing unchanged), so z then grows straight down the rows. Nodes whose four source neighbours
are not all valid are written as -1, the tifxyz "no node" value.

  --angle DEG       use this angle instead of the fitted one (degrees from the row axis, as printed)
  --flip-z          turn a further 180 degrees (z growing up the rows), for a scroll scanned upside down
  --center X Y      the scroll axis at the patch's height, in the patch's voxel frame
  --umbilicus FILE  {"control_points": [{"x": .., "y": .., "z": ..}, ...]}, interpolated in z, instead of --center
With --center or --umbilicus the grid normal N = dP/dU x dP/dV is also made to point OUT of the scroll, by reversing
the columns when it points in. Rows along +z with N outward is the standard segment orientation described next to
--flip-normals in villa's vc_render_tifxyz.cpp: rendered with --flip-normals the layers run towards a reader on the
scroll's centre side, the image is not mirrored, and one reading direction is enough.

meta.json is copied with a new uuid, the applied angle and the source uuid. generations.tif and other per-node extras
are not carried over (they no longer line up). Needs numpy, tifffile and opencv. Nothing is downloaded.
"""
import argparse
import json
import os
import uuid

import cv2
import numpy as np
import tifffile


def load(path):
    """x, y, z grids (float32) and the mask of valid nodes"""
    X, Y, Z = (tifffile.imread(os.path.join(path, f"{a}.tif")).astype(np.float32) for a in "xyz")
    return X, Y, Z, (Z > 0) & (X > 0) & (Y > 0)


def fitted_angle(Z, V):
    """direction of the z gradient across the grid, in degrees from the row axis, and its size in voxels per node"""
    r, c = np.nonzero(V)
    A = np.stack([r, c, np.ones(len(r))], 1).astype(np.float64)
    (a, b, _), *_ = np.linalg.lstsq(A, Z[V].astype(np.float64), rcond=None)
    return float(np.degrees(np.arctan2(b, a))), float(np.hypot(a, b))


def off_axis(angle):
    """degrees between the z gradient and the nearest grid axis (0 = level or a quarter turn from level)"""
    return abs(((angle + 45) % 90) - 45)


def level(X, Y, Z, V, angle):
    """resample the grid so that the direction `angle` becomes +row; returns (X, Y, Z) and the valid mask, trimmed"""
    h, w = Z.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    cs, sn = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(np.ceil(h * sn + w * cs)), int(np.ceil(h * cs + w * sn))
    M[0, 2] += W / 2 - w / 2
    M[1, 2] += H / 2 - h / 2
    # fill the holes with the nearest valid node first, so bilinear sampling never mixes a -1 into a coordinate
    _, lab = cv2.distanceTransformWithLabels((~V).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    vr, vc = np.nonzero(V)
    idx = np.zeros(lab.max() + 1, np.int64)
    idx[lab[vr, vc]] = np.arange(len(vr))
    near = idx[lab]
    out = []
    for A in (X, Y, Z):
        F = A[vr[near], vc[near]].astype(np.float32)
        out.append(cv2.warpAffine(F, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    ok = cv2.warpAffine(V.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    for A in out:
        A[~ok] = -1.0
    rr, cc = np.nonzero(ok)
    sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    return [A[sl] for A in out], ok[sl]


def outward_share(X, Y, Z, V, center):
    """share of grid cells whose normal dP/dU x dP/dV points away from the axis at `center` (x, y)"""
    G = np.stack([X, Y, Z], -1).astype(np.float64)
    c = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:]
    N = np.cross((G[:-1, 1:] - G[:-1, :-1])[c], (G[1:, :-1] - G[:-1, :-1])[c])
    return float(np.mean(((G[:-1, :-1][c][:, :2] - np.asarray(center, float)) * N[:, :2]).sum(1) > 0))


def axis_from_umbilicus(path, z):
    cp = sorted(json.load(open(path))["control_points"], key=lambda q: q["z"])
    return np.array([np.interp(z, [q["z"] for q in cp], [q[a] for q in cp]) for a in "xy"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Turn a tifxyz surface so that text rows come out level.")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--angle", type=float)
    ap.add_argument("--flip-z", action="store_true")
    ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--umbilicus")
    a = ap.parse_args(argv)
    X, Y, Z, V = load(a.src)
    fit, slope = fitted_angle(Z, V)
    ang = fit if a.angle is None else a.angle
    if a.flip_z:
        ang += 180.0
    (Xo, Yo, Zo), ok = level(X, Y, Z, V, ang)
    ctr, share, flipped = None, None, False
    if a.center is not None:
        ctr = np.array(a.center)
    elif a.umbilicus:
        ctr = axis_from_umbilicus(a.umbilicus, float(np.median(Zo[ok])))
    if ctr is not None:
        share = outward_share(Xo, Yo, Zo, ok, ctr)
        flipped = share < 0.5
        if flipped:
            Xo, Yo, Zo, ok = (np.ascontiguousarray(A[:, ::-1]) for A in (Xo, Yo, Zo, ok))
    os.makedirs(a.dst, exist_ok=False)
    for name, A in zip("xyz", (Xo, Yo, Zo)):
        tifffile.imwrite(os.path.join(a.dst, f"{name}.tif"), A.astype(np.float32))
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    meta.update(levelled_from=meta.get("uuid"), levelled_angle_deg=ang, levelled_columns_reversed=bool(flipped), uuid=str(uuid.uuid4()),
                bbox=[[float(A[ok].min()) for A in (Xo, Yo, Zo)], [float(A[ok].max()) for A in (Xo, Yo, Zo)]])
    json.dump(meta, open(os.path.join(a.dst, "meta.json"), "w"), indent=4)
    after, slope_after = fitted_angle(Zo, ok)
    print(f"{os.path.basename(os.path.normpath(a.src))}: grid {Z.shape} valid {int(V.sum())}; scroll axis at {fit:.1f} deg from the row axis "
          f"({off_axis(fit):.1f} deg off the nearest grid axis), {slope:.2f} voxels of z per node")
    if ctr is not None:
        print(f"  axis at ({ctr[0]:.0f}, {ctr[1]:.0f}): the normal pointed out of the scroll on {share:.2f} of the cells -> "
              f"columns {'reversed, it now points out' if flipped else 'kept'}")
    print(f"  -> {a.dst}: grid {Zo.shape} valid {int(ok.sum())} ({ok.sum() / V.sum():.3f} of the nodes kept); axis now at {after:.2f} deg, "
          f"{slope_after:.2f} voxels of z per node")


if __name__ == "__main__":
    main()
