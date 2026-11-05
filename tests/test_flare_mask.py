"""Tests for the VIIRS flare mask in plumescout.infer (TRD 5.6)."""

import numpy as np
import pandas as pd
from affine import Affine

from plumescout import infer

# 200 x 200 px, 10 m, UTM-like grid with origin (500000, 3210000)
TR = Affine(10, 0, 500000, 0, -10, 3210000)
T0 = pd.Timestamp("2023-04-05T07:30Z")


def _comp(r0, r1, c0, c1):
    m = np.zeros((200, 200), bool)
    m[r0:r1, c0:c1] = True
    return m


def test_fire_within_750m_masks_component() -> None:
    comp = _comp(100, 110, 100, 110)  # x 501000..501100, y 3208900..3209000
    fires = pd.DataFrame(
        {"x": [501100 + 700], "y": [3208950], "datetime_utc": [T0 - pd.Timedelta(hours=11)]}
    )
    assert infer.flare_flags([comp], TR, T0, fires, 750, pd.Timedelta(hours=12)) == [True]


def test_fire_beyond_750m_or_12h_does_not_mask() -> None:
    comp = _comp(100, 110, 100, 110)
    fires = pd.DataFrame(
        {
            "x": [501100 + 800, 501050],
            "y": [3208950, 3208950],
            "datetime_utc": [T0, T0 + pd.Timedelta(hours=13)],
        }
    )
    assert infer.flare_flags([comp], TR, T0, fires, 750, pd.Timedelta(hours=12)) == [False]


def test_flags_per_component() -> None:
    near = _comp(0, 5, 0, 5)  # top-left corner, x 500000..500050
    far = _comp(190, 200, 190, 200)  # bottom-right, ~2.7 km away
    fires = pd.DataFrame({"x": [500025], "y": [3209975], "datetime_utc": [T0]})
    flags = infer.flare_flags([near, far], TR, T0, fires, 750, pd.Timedelta(hours=12))
    assert flags == [True, False]


def test_no_fires() -> None:
    comp = _comp(0, 5, 0, 5)
    empty = pd.DataFrame(columns=["x", "y", "datetime_utc"])
    assert infer.flare_flags([comp], TR, T0, empty, 750, pd.Timedelta(hours=12)) == [False]


def test_chip_to_model_input_shapes_and_ranges() -> None:
    rng = np.random.default_rng(0)
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    model_bands = ["B11", "B12", "B2"]
    dn = rng.integers(1000, 4000, (13, 20, 20)).astype(np.uint16)
    ref = rng.integers(1000, 4000, (13, 20, 20)).astype(np.uint16)
    m = {
        "slope": {"B11": 287.49, "B12": 322.84, "B2": 466.17},
        "offset": {"B11": -2.15, "B12": -3.6, "B2": -48.78},
        "mbmp_stretch_pct": [1, 99],
    }
    x = infer.chip_to_model_input(dn, ref, bands, model_bands, m)
    assert x.shape == (4, 20, 20) and x.dtype == np.float32
    assert x.min() >= 0 and x.max() <= 1
    # band value: clip(slope * DN / 10000 + offset, 0, 255) / 255
    exp = np.clip(287.49 * dn[11, 3, 4] / 10000 - 2.15, 0, 255) / 255
    assert abs(x[0, 3, 4] - exp) < 1e-5
    # MBMP channel is stretched to [0, 1]
    assert x[3].min() == 0 and x[3].max() == 1


def test_chip_to_model_input_without_reference_gives_nan_free_mbmp() -> None:
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    dn = np.full((13, 8, 8), 2000, np.uint16)
    m = {"slope": {"B11": 287.49}, "offset": {"B11": -2.15}, "mbmp_stretch_pct": [1, 99]}
    x = infer.chip_to_model_input(dn, None, bands, ["B11"], m)
    assert np.isfinite(x).all() and x.shape == (2, 8, 8)
