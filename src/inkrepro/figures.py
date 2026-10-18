"""Render what two checkpoints disagree about.

The classification into "both / only A / only B" is always done at full
resolution and only then downsampled for display. Downsampling the
probabilities first and thresholding afterwards answers a different question:
a blurred 0.4 and 0.6 average to 0.5 and appear as ink that neither run
actually called.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# both runs called ink: the part of the reading that is reproducible
BOTH = (0.05, 0.05, 0.05)
# one run called ink and the other did not
ONLY_A = (0.84, 0.19, 0.19)
ONLY_B = (0.16, 0.38, 0.78)
# inside the valid region, neither called ink
NEITHER = (0.93, 0.93, 0.91)
# outside the valid region: nothing was asked
OUTSIDE = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Panels:
    """Display-sized arrays plus the counts they were summarised from."""

    a: np.ndarray
    b: np.ndarray
    agreement: np.ndarray
    n_both: int
    n_only_a: int
    n_only_b: int
    n_valid: int
    factor: int


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    """Average over factor x factor blocks, trimming any ragged edge."""

    if factor <= 1:
        return values.astype(np.float32)
    h = (values.shape[0] // factor) * factor
    w = (values.shape[1] // factor) * factor
    trimmed = values[:h, :w].astype(np.float32)
    return trimmed.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def _fit_factor(shape: tuple[int, int], target_long_edge: int) -> int:
    longest = max(shape)
    if longest <= target_long_edge:
        return 1
    return max(1, int(np.ceil(longest / target_long_edge)))


def build_panels(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    target_long_edge: int = 1600,
) -> Panels:
    """Classify at full resolution, then shrink for display."""

    valid = mask > 0
    ink_a = (a >= threshold) & valid
    ink_b = (b >= threshold) & valid

    both = ink_a & ink_b
    only_a = ink_a & ~ink_b
    only_b = ink_b & ~ink_a

    factor = _fit_factor(a.shape, target_long_edge)

    # Each class is reduced separately so the displayed colour is the true
    # local proportion of that class, not a threshold applied to a blur.
    f_both = _block_mean(both, factor)
    f_a = _block_mean(only_a, factor)
    f_b = _block_mean(only_b, factor)
    f_valid = _block_mean(valid, factor)
    f_none = np.clip(f_valid - f_both - f_a - f_b, 0.0, 1.0)
    f_out = np.clip(1.0 - f_valid, 0.0, 1.0)

    rgb = np.zeros((*f_both.shape, 3), dtype=np.float32)
    for weight, colour in (
        (f_both, BOTH),
        (f_a, ONLY_A),
        (f_b, ONLY_B),
        (f_none, NEITHER),
        (f_out, OUTSIDE),
    ):
        rgb += weight[..., None] * np.asarray(colour, dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    # ink dark on white, which is how these maps are normally read
    shown_a = 1.0 - _block_mean(np.where(valid, a, 0.0), factor)
    shown_b = 1.0 - _block_mean(np.where(valid, b, 0.0), factor)

    return Panels(
        a=shown_a,
        b=shown_b,
        agreement=rgb,
        n_both=int(both.sum()),
        n_only_a=int(only_a.sum()),
        n_only_b=int(only_b.sum()),
        n_valid=int(valid.sum()),
        factor=factor,
    )


def densest_ink_window(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    size: int,
    *,
    threshold: float = 0.5,
    stride: int | None = None,
) -> tuple[int, int]:
    """Locate the window with the most ink, not the most disagreement.

    Choosing the window where the two runs differ most would be picking the
    evidence to fit the claim. The window a reader would actually look at is
    the one with the most ink in it, so that is the one returned.
    """

    ink = ((a >= threshold) | (b >= threshold)) & (mask > 0)
    h, w = ink.shape
    size = min(size, h, w)
    stride = stride or max(size // 4, 1)
    best, by, bx = -1.0, 0, 0
    for y in range(0, max(h - size, 0) + 1, stride):
        for x in range(0, max(w - size, 0) + 1, stride):
            score = float(ink[y : y + size, x : x + size].mean())
            if score > best:
                best, by, bx = score, y, x
    return by, bx
