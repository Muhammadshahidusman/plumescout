"""Tests for plumescout.ch4net_index (recovering CH4Net image site/date by image matching)."""

import numpy as np
import pytest

from plumescout import ch4net_index as ci


def test_centre_crop_odd_and_even() -> None:
    a = np.arange(7 * 6).reshape(7, 6)
    c = ci.centre_crop(a, 4)
    assert c.shape == (4, 4)
    assert c[0, 0] == a[1, 1]


def test_centre_crop_too_small_raises() -> None:
    with pytest.raises(ValueError):
        ci.centre_crop(np.zeros((3, 3)), 4)


def test_assign_recovers_permutation_with_confidence() -> None:
    rng = np.random.default_rng(0)
    ours = rng.normal(size=(6, 50))
    perm = np.array([3, 0, 5, 1, 4, 2])
    theirs = ours[perm] + rng.normal(scale=0.3, size=(6, 50))
    sim = ci.similarity(ci.zrows(theirs), ci.zrows(ours))
    cols, conf = ci.assign(sim)
    assert (cols == perm).all()
    assert (conf > 0).all()


def test_assign_handles_more_dates_than_images() -> None:
    rng = np.random.default_rng(1)
    ours = rng.normal(size=(8, 40))
    theirs = ours[[6, 2, 4]] + rng.normal(scale=0.2, size=(3, 40))
    cols, conf = ci.assign(ci.similarity(ci.zrows(theirs), ci.zrows(ours)))
    assert list(cols) == [6, 2, 4]
    assert conf.shape == (3,)


def test_place_label_on_chip_centres_and_pads() -> None:
    lab = np.zeros((5, 3))
    lab[2, 1] = 1  # centre pixel
    out = ci.place_centred(lab, (9, 9))
    assert out.shape == (9, 9)
    assert out[4, 4] == 1 and out.sum() == 1


def test_place_label_on_chip_crops_larger_label() -> None:
    lab = np.zeros((11, 11))
    lab[5, 5] = 1
    lab[0, 0] = 1  # falls outside after crop
    out = ci.place_centred(lab, (5, 5))
    assert out[2, 2] == 1 and out.sum() == 1
