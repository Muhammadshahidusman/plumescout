"""Tests for plumescout.mbmp (Varon et al. 2021 multi-band methods).

Sign convention: methane absorbs in B12, so plume pixels have dR < 0.
"""

import numpy as np
import pytest

from plumescout import mbmp

SHAPE = (200, 200)


def _surface(seed: int = 0, noise: float = 0.002) -> np.ndarray:
    """Smooth synthetic B11 reflectance field (unitless TOA reflectance)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    base = 0.30 + 0.05 * np.sin(xx / 25.0) + 0.03 * np.cos(yy / 40.0)
    return base + rng.normal(0.0, noise, SHAPE)


def _plume_mask(center=(100, 100), radius: int = 15) -> np.ndarray:
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def _r12_from(r11: np.ndarray, seed: int, ratio: float = 1.25, noise: float = 0.002) -> np.ndarray:
    """B12 = B11 / ratio plus independent sensor noise (bands are not perfectly correlated)."""
    return r11 / ratio + np.random.default_rng(seed + 100).normal(0.0, noise, r11.shape)


def test_fit_c_recovers_scale() -> None:
    r12 = _surface(1)
    r11 = 1.3 * r12
    assert mbmp.fit_c(r11, r12) == pytest.approx(1.3, rel=1e-6)


def test_single_pass_zero_without_plume() -> None:
    r12 = _surface(2, noise=0.0)
    r11 = 1.2 * r12
    dr = mbmp.single_pass(r11, r12)
    assert np.allclose(dr, 0.0, atol=1e-9)


def test_single_pass_negative_inside_injected_b12_absorption() -> None:
    r11 = _surface(3)
    r12 = _r12_from(r11, 3)
    plume = _plume_mask()
    r12 = np.where(plume, r12 * 0.90, r12)  # 10% B12 absorption
    dr = mbmp.single_pass(r11, r12)
    assert dr.shape == SHAPE
    assert np.median(dr[plume]) < -0.05
    assert abs(np.median(dr[~plume])) < 0.01
    assert np.median(dr[plume]) < np.median(dr[~plume])


def test_multi_pass_cancels_static_surface_feature() -> None:
    """A bright patch with a different B11/B12 ratio appears in both dates; plume only in target."""
    r11 = _surface(4)
    r12 = _r12_from(r11, 4)
    patch = _plume_mask(center=(50, 150), radius=20)
    r12_patch = np.where(patch, r12 * 0.85, r12)  # surface spectral feature, both dates

    ref11, ref12 = r11.copy(), r12_patch.copy()
    plume = _plume_mask(center=(140, 60), radius=15)
    tgt11, tgt12 = r11.copy(), np.where(plume, r12_patch * 0.90, r12_patch)

    single = mbmp.single_pass(tgt11, tgt12)
    multi = mbmp.multi_pass(tgt11, tgt12, ref11, ref12)

    assert np.median(single[patch]) < -0.05  # single-pass confuses surface with plume
    assert abs(np.median(multi[patch])) < 0.01  # multi-pass removes it
    assert np.median(multi[plume]) < -0.05  # and keeps the plume


def test_threshold_mask_finds_plume() -> None:
    r11 = _surface(5)
    r12 = _r12_from(r11, 5)
    plume = _plume_mask()
    r12 = np.where(plume, r12 * 0.90, r12)
    dr = mbmp.single_pass(r11, r12)
    mask = mbmp.threshold_mask(dr, k=3.0)
    assert mask.dtype == bool
    recall = (mask & plume).sum() / plume.sum()
    false_pos_rate = (mask & ~plume).sum() / (~plume).sum()
    assert recall > 0.9
    assert false_pos_rate < 0.01


def test_threshold_mask_ignores_positive_anomalies() -> None:
    dr = np.zeros(SHAPE)
    dr[10:20, 10:20] = 0.2  # positive, not methane
    assert not mbmp.threshold_mask(dr, k=3.0).any()


def test_invalid_pixels_give_nan_and_do_not_bias_fit() -> None:
    r12 = _surface(6)
    r11 = 1.3 * r12
    r11[0, :10] = np.nan
    r11[1, :10] = 0.0
    r12[2, :10] = np.nan
    assert mbmp.fit_c(r11, r12) == pytest.approx(1.3, rel=1e-6)
    dr = mbmp.single_pass(r11, r12)
    assert np.isnan(dr[0, :10]).all()
    assert np.isnan(dr[1, :10]).all()
    assert np.isnan(dr[2, :10]).all()
    assert not mbmp.threshold_mask(dr, k=3.0)[:3, :10].any()


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mbmp.single_pass(np.ones((5, 5)), np.ones((5, 4)))


def test_fit_c_raises_without_valid_pixels() -> None:
    with pytest.raises(ValueError):
        mbmp.fit_c(np.full((3, 3), np.nan), np.ones((3, 3)))
