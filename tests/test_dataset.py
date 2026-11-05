"""Tests for plumescout.dataset (CH4Net arrays -> model tensors)."""

import matplotlib
import numpy as np
import pytest

from plumescout import dataset as ds


def test_decode_magma_round_trip() -> None:
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 256, size=(30, 20))
    lut = (
        (matplotlib.colormaps["magma"](np.arange(256) / 255)[:, :3] * 255).round().astype(np.uint8)
    )
    rgba = np.concatenate([lut[vals], np.full((30, 20, 1), 255, np.uint8)], axis=-1)
    out = ds.decode_magma(rgba)
    assert out.shape == (30, 20) and out.dtype == np.float32
    assert np.abs(out - vals / 255).max() <= 1 / 255 + 1e-6


@pytest.mark.parametrize("seed", range(6))
def test_augment_only_moves_pixels(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.random((7, 16, 16)).astype(np.float32)
    y = (rng.random((16, 16)) > 0.8).astype(np.float32)
    xa, ya = ds.augment(x, y, rng)
    # same multiset of values per channel: no colour or brightness change (hard rule 6)
    for c in range(7):
        assert np.array_equal(np.sort(xa[c].ravel()), np.sort(x[c].ravel()))
    assert ya.sum() == y.sum()


def test_augment_keeps_input_and_label_aligned() -> None:
    rng = np.random.default_rng(3)
    x = np.zeros((7, 8, 8), np.float32)
    x[:, 1, 2] = 1.0
    y = np.zeros((8, 8), np.float32)
    y[1, 2] = 1.0
    for _ in range(10):
        xa, ya = ds.augment(x, y, rng)
        assert np.array_equal(xa[0] > 0, ya > 0)


def test_pad_to_and_valid_mask() -> None:
    x = np.ones((7, 217, 180), np.float32)
    xp, valid = ds.pad_to(x, 256, 192)
    assert xp.shape == (7, 256, 192)
    assert valid.sum() == 217 * 180
    assert xp[:, ~valid].sum() == 0


def test_random_crop_in_bounds() -> None:
    rng = np.random.default_rng(0)
    x = np.arange(7 * 20 * 18, dtype=np.float32).reshape(7, 20, 18)
    y = np.zeros((20, 18), np.float32)
    xc, yc = ds.random_crop(x, y, 16, rng)
    assert xc.shape == (7, 16, 16) and yc.shape == (16, 16)
