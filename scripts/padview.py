"""padview.py <render.zarr> <view.zarr>: zero-copy padded view of a render for inference.
Writes a zarr v2 group whose level-0 array declares Y and X rounded up to a multiple of 64 and symlinks the original
chunk directory; chunks beyond the original extent do not exist, so zarr returns fill_value 0 there. The strip beyond the
original shape inside the last stored chunk is read back and checked to be zero; if it is not, a real zero-padded copy is
written instead (zstd). Prints JSON with the shapes and which path was taken."""
import json, os, shutil, sys, numpy as np, zarr
src, dst = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(src, "0", ".zarray")))
Z, H, W = meta["shape"]; Hp, Wp = -(-H // 64) * 64, -(-W // 64) * 64
if os.path.lexists(dst): shutil.rmtree(dst)
os.makedirs(os.path.join(dst, "0"))
json.dump({"zarr_format": 2}, open(os.path.join(dst, ".zgroup"), "w"))
m = dict(meta); m["shape"] = [Z, Hp, Wp]; json.dump(m, open(os.path.join(dst, "0", ".zarray"), "w"))
os.symlink(os.path.abspath(os.path.join(src, "0", "0")), os.path.join(dst, "0", "0"))
a = zarr.open(dst, mode="r")["0"]; mode = "view"
ok = True
if Hp > H: ok &= not a[:, H:Hp, :].any()
if Wp > W: ok &= not a[:, :, W:Wp].any()
if not ok:
    shutil.rmtree(dst); o = zarr.open(src, mode="r")["0"][:]
    g = zarr.open_group(dst, mode="w", zarr_format=2)
    d = g.create_array("0", shape=(Z, Hp, Wp), chunks=(Z, 128, 128), dtype=o.dtype); d[:, :H, :W] = o; mode = "copy"
print(json.dumps({"shape": [Z, H, W], "padded": [Z, Hp, Wp], "mode": mode}))
