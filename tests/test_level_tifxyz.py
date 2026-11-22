"""level_tifxyz.py on a synthetic patch: a tilted square of a cylinder, like a patch grown from a seed."""
import importlib.util
import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
tifffile = pytest.importorskip("tifffile")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("level_tifxyz", os.path.join(HERE, "..", "levelling", "level_tifxyz.py"))
lt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lt)

CENTER = (4000.0, 4000.0)
RADIUS, STEP = 1500.0, 20.0


def make_patch(path, tilt_deg, n=120, hole=True, inward=False):
    """a square grid on a cylinder around CENTER (axis = z), its rows turned tilt_deg away from the z direction"""
    t = np.radians(tilt_deg)
    r, c = np.mgrid[0:n, 0:n].astype(np.float64)
    r -= n / 2
    c -= n / 2
    along_z = (r * np.cos(t) + c * np.sin(t)) * STEP          # distance along the scroll axis
    around = (-r * np.sin(t) + c * np.cos(t)) * STEP          # distance around the scroll
    if inward:
        around = -around                                       # mirror the grid: its normal now points the other way
    phi = around / RADIUS
    X = CENTER[0] + RADIUS * np.cos(phi)
    Y = CENTER[1] + RADIUS * np.sin(phi)
    Z = 9000.0 + along_z
    if hole:
        X[40:50, 60:75] = Y[40:50, 60:75] = Z[40:50, 60:75] = -1.0
    os.makedirs(path)
    for name, A in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(os.path.join(path, f"{name}.tif"), A.astype(np.float32))
    json.dump({"format": "tifxyz", "type": "seg", "scale": [0.05, 0.05], "uuid": "synthetic"}, open(os.path.join(path, "meta.json"), "w"))
    return path


def spacing(X, Y, Z, V):
    G = np.stack([X, Y, Z], -1).astype(np.float64)
    rows = np.linalg.norm(G[1:] - G[:-1], axis=-1)[V[1:] & V[:-1]]
    cols = np.linalg.norm(G[:, 1:] - G[:, :-1], axis=-1)[V[:, 1:] & V[:, :-1]]
    return float(np.median(rows)), float(np.median(cols))


@pytest.mark.parametrize("tilt", [0.0, 17.0, 43.0, -60.0, 135.0])
def test_rows_come_out_level(tmp_path, tilt):
    src = make_patch(str(tmp_path / "src"), tilt)
    dst = str(tmp_path / "dst")
    lt.main([src, dst])
    X, Y, Z, V = lt.load(dst)
    angle, slope = lt.fitted_angle(Z, V)
    assert abs(angle) < 0.5                                   # z grows straight down the rows
    assert slope == pytest.approx(STEP, rel=0.02)             # one node is still one 20-voxel step along z
    dr, dc = spacing(X, Y, Z, V)
    assert dr == pytest.approx(STEP, rel=0.02) and dc == pytest.approx(STEP, rel=0.02)
    # every levelled node still lies on the cylinder the patch was cut from
    rad = np.hypot(X[V] - CENTER[0], Y[V] - CENTER[1])
    assert np.abs(rad - RADIUS).max() < 0.5


def test_holes_stay_holes_and_no_minus_one_leaks_into_coordinates(tmp_path):
    src = make_patch(str(tmp_path / "src"), 30.0, hole=True)
    dst = str(tmp_path / "dst")
    lt.main([src, dst])
    X0, Y0, Z0, V0 = lt.load(src)
    X, Y, Z, V = lt.load(dst)
    assert 0.95 < V.sum() / V0.sum() < 1.0                    # the hole and a one-node rim are dropped, nothing else
    assert (~V).sum() > 0
    raw = [tifffile.imread(os.path.join(dst, f"{a}.tif")) for a in "xyz"]
    assert all(np.all(A[~V] == -1.0) for A in raw)
    assert Z[V].min() >= Z0[V0].min() - 1 and Z[V].max() <= Z0[V0].max() + 1   # no coordinate was mixed with a -1


def test_center_makes_the_normal_point_out(tmp_path):
    for name, inward in (("out", False), ("in", True)):
        src = make_patch(str(tmp_path / f"src_{name}"), 25.0, hole=False, inward=inward)
        X0, Y0, Z0, V0 = lt.load(src)
        before = lt.outward_share(X0, Y0, Z0, V0, CENTER)
        dst = str(tmp_path / f"dst_{name}")
        lt.main([src, dst, "--center", str(CENTER[0]), str(CENTER[1])])
        X, Y, Z, V = lt.load(dst)
        assert lt.outward_share(X, Y, Z, V, CENTER) > 0.99
        meta = json.load(open(os.path.join(dst, "meta.json")))
        assert meta["levelled_columns_reversed"] == (before < 0.5)
        assert abs(lt.fitted_angle(Z, V)[0]) < 0.5
    # the two starting grids were mirror images of each other, so exactly one of them had to be reversed
    a = json.load(open(str(tmp_path / "dst_out" / "meta.json")))["levelled_columns_reversed"]
    b = json.load(open(str(tmp_path / "dst_in" / "meta.json")))["levelled_columns_reversed"]
    assert a != b


def test_angle_override_and_meta(tmp_path):
    src = make_patch(str(tmp_path / "src"), 40.0, hole=False)
    X0, Y0, Z0, V0 = lt.load(src)
    fit, _ = lt.fitted_angle(Z0, V0)
    dst = str(tmp_path / "dst")
    lt.main([src, dst, "--angle", str(fit + 15.0)])
    X, Y, Z, V = lt.load(dst)
    assert abs(abs(lt.fitted_angle(Z, V)[0]) - 15.0) < 0.5    # left 15 degrees off level on purpose
    meta = json.load(open(os.path.join(dst, "meta.json")))
    assert meta["levelled_from"] == "synthetic" and meta["uuid"] != "synthetic"
    assert meta["levelled_angle_deg"] == pytest.approx(fit + 15.0)
    with pytest.raises(FileExistsError):
        lt.main([src, dst])                                   # never overwrites an existing surface
