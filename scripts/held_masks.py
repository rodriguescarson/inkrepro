"""The exact atlas valid mask for the four held-out PHerc0139 windings.

The atlas defines a surface's valid mask as

    erode(max over z of (volume > 0), iterations=64, border_value=0)

(eligible-scroll-atlas/scripts/infer_v2.sh). inkrepro's held-out result currently
reconstructs that from the ink maps' own footprint, because w051/w055/w058/w059
ship no valid_mask.tif. Measured against published masks on eight atlas surfaces in
the same coverage band, the reconstruction errs by up to 0.047 IoU -- about half the
smallest effect it is used to support. This computes the real thing from the
published 9.362 um surface volume.

The volumes are uncompressed uint8 zarr v2 in us-east-1, chunked [28,128,128] with
fill_value 0, so a chunk that is absent is a chunk of zeros. Every chunk is collapsed
to its max over z the moment it arrives and then dropped: peak memory is one boolean
canvas per surface (~73 MB), and nothing is written to disk until the mask itself.
"""

import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from scipy import ndimage

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOL = "9.362um-1.2m-113keV-volume-20250728140407.zarr"
CZ, CY, CX = 28, 128, 128
CHUNK_BYTES = CZ * CY * CX
ITERATIONS = 64

SEGS = {
    "w051": "20260306000001-w051_2026030600",
    "w055": "20251226000000-w055_2025122611",
    "w058": "20260210000000-w058_2026021020",
    "w059": "20250223000000-w059_2025022312",
}

_local = threading.local()


def session():
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=3)
        s.mount("https://", a)
        _local.s = s
    return s


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def fetch_chunk(args):
    """Return (row, col, 128x128 bool) -- the chunk's max over z, or None if absent."""
    base, r, c = args
    url = f"{base}/0/{r}/{c}"
    resp = session().get(url, timeout=90)
    if resp.status_code == 404:
        return r, c, None            # never written: fill_value 0 everywhere
    resp.raise_for_status()
    buf = resp.content
    if len(buf) != CHUNK_BYTES:      # a short chunk is a truncated read, not a shape
        raise ValueError(f"{url}: {len(buf)} bytes, expected {CHUNK_BYTES}")
    a = np.frombuffer(buf, dtype=np.uint8).reshape(CZ, CY, CX)
    return r, c, a.max(axis=0) > 0


def footprint(seg, shape, workers=48):
    """max over z of (volume > 0), streamed."""
    Z, H, W = shape
    base = f"{BUCKET}/PHerc0139/segments/{seg}/surface-volumes/{VOL}/0"
    nr, nc = -(-H // CY), -(-W // CX)
    fp = np.zeros((nr * CY, nc * CX), dtype=bool)   # chunk-aligned, cropped at the end
    jobs = [(base, r, c) for r in range(nr) for c in range(nc)]
    done = absent = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, c, block in ex.map(fetch_chunk, jobs):
            done += 1
            if block is None:
                absent += 1
            else:
                fp[r * CY:(r + 1) * CY, c * CX:(c + 1) * CX] = block
            if done % 500 == 0:
                rate = done * CHUNK_BYTES / 1e6 / (time.time() - t0)
                log(f"  {done}/{len(jobs)} chunks  {rate:.0f} MB/s")
    log(f"  {done} chunks, {absent} absent, {time.time()-t0:.0f}s")
    return fp[:H, :W]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/out"
    os.makedirs(out, exist_ok=True)
    report = {}
    for w, seg in SEGS.items():
        log(f"=== {w}  {seg}")
        meta = session().get(f"{BUCKET}/PHerc0139/segments/{seg}/surface-volumes/{VOL}/0/.zarray",
                             timeout=60).json()
        shape = tuple(meta["shape"])
        assert tuple(meta["chunks"]) == (CZ, CY, CX), meta["chunks"]
        assert meta["compressor"] is None and meta["dtype"] == "|u1", meta

        fp = footprint(seg, shape)
        log(f"  footprint {int(fp.sum()):,} px of {fp.size:,}")

        t0 = time.time()
        mask = ndimage.binary_erosion(fp, iterations=ITERATIONS, border_value=0)
        log(f"  eroded in {time.time()-t0:.0f}s -> {int(mask.sum()):,} px")

        # Control: binary_erosion's default structure is 4-connected, so the same
        # mask must fall out of a taxicab distance transform. Pad first, because
        # border_value=0 means the world outside the canvas counts as background.
        pad = ITERATIONS + 1
        padded = np.pad(fp, pad, mode="constant", constant_values=False)
        cdt = ndimage.distance_transform_cdt(padded, metric="taxicab")[pad:-pad, pad:-pad]
        agree = bool(np.array_equal(cdt > ITERATIONS, mask))
        log(f"  distance-transform cross-check: {'agrees' if agree else 'DISAGREES'}")

        np.savez_compressed(os.path.join(out, f"{w}_mask.npz"),
                            packed=np.packbits(mask), shape=np.array(mask.shape))
        report[w] = {
            "segment": seg, "shape": list(shape),
            "footprint_px": int(fp.sum()), "mask_px": int(mask.sum()),
            "mask_over_footprint": round(float(mask.sum()) / float(fp.sum()), 6),
            "erosion_cross_check_agrees": agree,
        }
        log(f"  {json.dumps(report[w])}")

    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
