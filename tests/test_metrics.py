"""The evaluator is validated before it is used to rank anything.

An A/A test (a prediction against itself) must score a perfect 1.0, and a
planted perturbation of known size must move the score in the known direction.
A metric that has not passed both cannot tell a real disagreement from its own
arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from inkrepro.metrics import agreement, align, to_unit, valid_mask


def _pred(seed: int, shape=(128, 128)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


# --- A/A: the evaluator must not invent disagreement ----------------------


def test_identical_predictions_score_perfectly():
    a = _pred(0)
    got = agreement(a, a.copy())
    assert got.pearson_r == pytest.approx(1.0)
    assert got.iou == pytest.approx(1.0)
    assert got.dice == pytest.approx(1.0)
    assert got.mean_abs_diff == pytest.approx(0.0)
    assert got.frac_ink_a == pytest.approx(got.frac_ink_b)


def test_aa_holds_after_uint8_round_trip():
    """Stored predictions are uint8; quantisation must not break the A/A."""
    a = (_pred(1) * 255).astype(np.uint8)
    unit = to_unit(a)
    got = agreement(unit, unit)
    assert got.iou == pytest.approx(1.0)
    assert got.pearson_r == pytest.approx(1.0)


# --- planted perturbations of known size ----------------------------------


def test_agreement_falls_monotonically_as_noise_is_planted():
    base = _pred(2)
    scores = []
    for sigma in (0.0, 0.05, 0.15, 0.4):
        rng = np.random.default_rng(99)
        noised = np.clip(base + rng.normal(0, sigma, base.shape), 0, 1)
        scores.append(agreement(base, noised.astype(np.float32)).pearson_r)
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] == pytest.approx(1.0)
    assert scores[-1] < 0.8


def test_planted_label_flip_lands_on_the_arithmetic_iou():
    """Flip a known number of ink pixels; IoU must match the hand computation."""
    a = np.zeros((10, 10), dtype=np.float32)
    a[:, :4] = 1.0  # 40 ink pixels
    b = a.copy()
    b[:, 3] = 0.0  # remove 10 of them -> 30 shared, 40 union
    got = agreement(a, b, mask=np.ones_like(a))
    assert got.iou == pytest.approx(30 / 40)
    assert got.dice == pytest.approx(2 * 30 / (40 + 30))


def test_disjoint_ink_scores_zero_overlap():
    a = np.zeros((8, 8), dtype=np.float32)
    b = np.zeros((8, 8), dtype=np.float32)
    a[:4] = 1.0
    b[4:] = 1.0
    got = agreement(a, b, mask=np.ones_like(a))
    assert got.iou == pytest.approx(0.0)
    assert got.dice == pytest.approx(0.0)


def test_no_ink_anywhere_is_agreement_not_division_by_zero():
    a = np.full((8, 8), 0.1, dtype=np.float32)
    b = np.full((8, 8), 0.2, dtype=np.float32)
    got = agreement(a, b, mask=np.ones_like(a))
    assert got.iou == pytest.approx(1.0)
    assert got.dice == pytest.approx(1.0)


# --- the masking rule, which is where the number is easiest to inflate ----


def test_background_pixels_inflate_the_correlation():
    """Two runs that anticorrelate on every real pixel must not look like they
    agree because they both wrote zero across the empty 80% of the canvas."""
    a = np.zeros((10, 10), dtype=np.float32)
    b = np.zeros((10, 10), dtype=np.float32)
    # the traced surface is the first two rows; the runs disagree across it
    ramp = np.linspace(0.05, 0.95, 20, dtype=np.float32).reshape(2, 10)
    a[:2] = ramp
    b[:2] = ramp[::-1, ::-1]

    masked = agreement(a, b)
    assert masked.n_valid == 20
    assert masked.pearson_r < 0.0  # they genuinely anticorrelate

    unmasked = agreement(a, b, mask=np.ones_like(a))
    assert unmasked.n_valid == 100
    # 80 shared zeros drag a negative correlation up above zero
    assert unmasked.pearson_r > masked.pearson_r
    assert unmasked.pearson_r > 0.0


def test_explicit_mask_further_restricts_the_fallback():
    a = np.full((4, 4), 0.8, dtype=np.float32)
    b = np.full((4, 4), 0.9, dtype=np.float32)
    explicit = np.zeros((4, 4), dtype=np.uint8)
    explicit[:2] = 1
    assert agreement(a, b).n_valid == 16
    assert agreement(a, b, mask=explicit).n_valid == 8


def test_constant_map_gives_nan_correlation_not_a_fake_zero():
    a = np.full((8, 8), 0.5, dtype=np.float32)
    b = _pred(3, (8, 8))
    assert np.isnan(agreement(a, b, mask=np.ones_like(a)).pearson_r)


def test_shape_mismatch_is_an_error_not_a_silent_crop():
    with pytest.raises(ValueError, match="differ in shape"):
        valid_mask(np.zeros((4, 4)), np.zeros((4, 5)))


def test_empty_mask_is_an_error():
    a = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="no valid pixels"):
        agreement(a, a)


# --- storage conventions ---------------------------------------------------


def test_to_unit_scales_by_dtype_range_not_by_observed_max():
    dim = np.full((4, 4), 100, dtype=np.uint8)
    assert to_unit(dim).max() == pytest.approx(100 / 255)
    with pytest.raises(TypeError):
        to_unit(np.zeros((2, 2), dtype=np.int32))


def test_align_crops_padded_canvases_to_the_shared_origin():
    big = np.ones((2304, 10624), dtype=np.uint8)
    small = np.ones((2300, 10600), dtype=np.uint8)
    out = align(big, small)
    assert all(o.shape == (2300, 10600) for o in out)


def test_constant_float32_region_is_nan_not_a_divide_by_almost_zero():
    """np.full(..., 0.1, float32).std() is 7.45e-09, not 0. An `== 0` guard
    lets that reach corrcoef and return a meaningless number with a warning."""
    a = np.full((8, 8), 0.1, dtype=np.float32)
    b = np.full((8, 8), 0.2, dtype=np.float32)
    assert a.std() != 0  # the precision trap this guards against
    with np.errstate(all="raise"):
        got = agreement(a, b, mask=np.ones_like(a))
    assert np.isnan(got.pearson_r)


def test_one_uint8_step_still_counts_as_real_variation():
    """The constant tolerance must not swallow the smallest storable change."""
    a = np.full((16, 16), 100, dtype=np.uint8)
    a.flat[0] = 101
    b = np.full((16, 16), 100, dtype=np.uint8)
    b.flat[0] = 101
    got = agreement(to_unit(a), to_unit(b), mask=np.ones((16, 16), dtype=np.uint8))
    assert not np.isnan(got.pearson_r)
    assert got.pearson_r == pytest.approx(1.0)
