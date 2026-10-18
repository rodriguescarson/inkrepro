"""Agreement between two ink predictions of the same surface.

Every metric here is computed over a valid-pixel mask and nothing else. The
rendered surface volumes are zero outside the traced sheet, and the ink models
write zero there too, so including those pixels inflates every agreement score
with pixels on which the two models were never asked anything.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# Far below one uint8 step (1/255 = 3.9e-3), so it only catches true constants.
_CONSTANT_TOL = 1e-9


@dataclass(frozen=True)
class Agreement:
    """How much two predictions of one surface say the same thing."""

    n_valid: int
    pearson_r: float
    iou: float
    dice: float
    frac_ink_a: float
    frac_ink_b: float
    mean_abs_diff: float
    threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def valid_mask(
    a: np.ndarray,
    b: np.ndarray,
    explicit: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels on which both predictions carry a real answer.

    An explicit mask (the ``valid_mask.tif`` written alongside a run) is
    authoritative and is used on its own. It is *not* intersected with
    ``a > 0``, because a pixel inside the valid region that a model scored
    exactly 0.0 is a real prediction of "no ink" and dropping it would bias the
    comparison toward the pixels both models already felt strongly about.

    Without an explicit mask the fallback is ``a > 0 and b > 0``, which is all
    the stored TIFFs make available. That fallback does discard genuine zeros,
    so it is a convenience, not an equivalent. Prefer the explicit mask.
    """

    if a.shape != b.shape:
        raise ValueError(f"predictions differ in shape: {a.shape} vs {b.shape}")
    if explicit is not None:
        if explicit.shape != a.shape:
            raise ValueError(
                f"mask shape {explicit.shape} does not match predictions {a.shape}"
            )
        return explicit > 0
    return (a > 0) & (b > 0)


def agreement(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Agreement:
    """Compare two ink probability maps scaled to [0, 1].

    ``iou`` is the number worth reading: of every pixel that at least one of the
    two runs called ink, the fraction that both called ink. ``pearson_r`` is
    reported because it is what the literature quotes, but it is computed over
    mostly-non-ink pixels and so reads higher than the ink actually agrees.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = valid_mask(a, b, mask)
    n = int(m.sum())
    if n == 0:
        raise ValueError("no valid pixels to compare")

    # float64 for the statistics: the std of a constant float32 array is not
    # exactly zero (np.full((8, 8), 0.1, np.float32).std() is 7.45e-09), so an
    # `== 0` guard in float32 lets a constant region reach corrcoef and come
    # back as a divide-by-almost-zero instead of a clean NaN.
    av = a[m].astype(np.float64)
    bv = b[m].astype(np.float64)

    # A constant map has no variance, so Pearson r is undefined rather than 0.
    # The tolerance sits far below one uint8 step (1/255 = 3.9e-3), so it can
    # never swallow real variation.
    if av.std() <= _CONSTANT_TOL or bv.std() <= _CONSTANT_TOL:
        r = float("nan")
    else:
        r = float(np.corrcoef(av, bv)[0, 1])

    ink_a = av >= threshold
    ink_b = bv >= threshold
    union = int((ink_a | ink_b).sum())
    inter = int((ink_a & ink_b).sum())
    # No ink called by either run: the two agree completely about its absence,
    # which is 1.0 rather than 0/0.
    iou = 1.0 if union == 0 else inter / union
    denom = int(ink_a.sum()) + int(ink_b.sum())
    dice = 1.0 if denom == 0 else 2.0 * inter / denom

    return Agreement(
        n_valid=n,
        pearson_r=r,
        iou=float(iou),
        dice=float(dice),
        frac_ink_a=float(ink_a.mean()),
        frac_ink_b=float(ink_b.mean()),
        mean_abs_diff=float(np.abs(av - bv).mean()),
        threshold=float(threshold),
    )


def to_unit(image: np.ndarray) -> np.ndarray:
    """Scale a stored prediction to [0, 1].

    uint8 and uint16 TIFFs are divided by their full range. Float input is
    returned unchanged and is assumed to already be a probability; it is not
    rescaled, because rescaling by the observed max would make the result
    depend on the brightest pixel in the crop.
    """

    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    raise TypeError(f"unsupported prediction dtype {arr.dtype}")


def align(*images: np.ndarray) -> list[np.ndarray]:
    """Crop images to their common top-left overlap.

    Folder-mode inference pads the canvas to a multiple of the patch size, so a
    prediction can come back a few pixels larger than the mask written beside
    it. Cropping is correct only because every run shares the origin.
    """

    h = min(int(i.shape[0]) for i in images)
    w = min(int(i.shape[1]) for i in images)
    return [np.asarray(i)[:h, :w] for i in images]
