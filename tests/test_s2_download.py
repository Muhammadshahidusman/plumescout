"""Tests for the pure (non-GEE) parts of plumescout.s2_download."""

from datetime import date

import numpy as np
import pytest

from plumescout import s2_download as s2


@pytest.mark.parametrize(
    ("lon", "lat", "epsg"),
    [
        (53.779, 39.460, 32639),  # Turkmenistan T7, zone 39N
        (47.4, 30.6, 32638),  # Basra, zone 38N
        (-0.5, 51.5, 32630),
        (151.2, -33.9, 32756),  # southern hemisphere
        (4.0, 60.0, 32632),  # Norway exception: 4E 60N is zone 32V, not 31
    ],
)
def test_utm_epsg(lon: float, lat: float, epsg: int) -> None:
    assert s2.utm_epsg(lon, lat) == epsg


def test_chip_grid_is_centred_200px_10m() -> None:
    g = s2.chip_grid(53.779, 39.460, size_px=200, pixel_m=10)
    assert g.epsg == 32639
    assert g.width == g.height == 200
    assert g.transform[0] == 10 and g.transform[4] == -10
    # centre of the grid maps back to the site within one pixel
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, g.epsg, always_xy=True).transform(53.779, 39.460)
    cx = g.transform[2] + 100 * 10
    cy = g.transform[5] - 100 * 10
    assert abs(cx - x) <= 10 and abs(cy - y) <= 10
    # origin snapped to the 10 m pixel grid
    assert g.transform[2] % 10 == 0 and g.transform[5] % 10 == 0


def test_pick_reference_most_recent_clear_within_lookback() -> None:
    scenes = [
        (date(2021, 5, 1), 0.05),
        (date(2021, 5, 11), 0.01),
        (date(2021, 5, 16), 0.60),  # cloudy
        (date(2021, 5, 21), 0.00),  # target itself
        (date(2021, 5, 26), 0.00),  # after target
    ]
    ref = s2.pick_reference(scenes, date(2021, 5, 21), lookback_days=30, max_cloud=0.2)
    assert ref == date(2021, 5, 11)


def test_pick_reference_none_when_nothing_clear_in_window() -> None:
    scenes = [(date(2021, 1, 1), 0.0), (date(2021, 3, 1), 0.9)]
    assert s2.pick_reference(scenes, date(2021, 3, 6), lookback_days=30, max_cloud=0.2) is None


def test_pick_reference_lookback_is_inclusive() -> None:
    scenes = [(date(2021, 2, 4), 0.0)]
    assert s2.pick_reference(scenes, date(2021, 3, 6), 30, 0.2) == date(2021, 2, 4)


def test_cloud_fraction() -> None:
    prob = np.zeros((10, 10))
    prob[:2, :] = 80  # 20 cloudy pixels
    prob[5, 5] = np.nan  # no data, excluded
    assert s2.cloud_fraction(prob, threshold=40) == pytest.approx(20 / 99)


def test_cloud_fraction_all_nan_is_one() -> None:
    assert s2.cloud_fraction(np.full((3, 3), np.nan), threshold=40) == 1.0


def test_shadow_fraction_flags_dark_nir() -> None:
    median = np.full((10, 10), 0.30)
    nir = median.copy()
    nir[:3, :] = 0.10  # 30 shadowed pixels
    nir[9, 9] = np.nan
    assert s2.shadow_fraction(nir, median, ratio=0.6) == pytest.approx(30 / 99)


def test_shadow_fraction_ignores_mild_darkening() -> None:
    median = np.full((5, 5), 0.30)
    assert s2.shadow_fraction(median * 0.8, median, ratio=0.6) == 0.0


def test_obscured_combines_and_caps() -> None:
    assert s2.obscured_fraction(0.1, 0.15) == pytest.approx(0.25)
    assert s2.obscured_fraction(0.8, 0.5) == 1.0


def test_reference_candidates_newest_first_within_lookback() -> None:
    avail = [
        date(2023, 3, 1),
        date(2023, 3, 6),
        date(2023, 3, 20),
        date(2023, 4, 4),
        date(2023, 4, 5),
    ]
    out = s2.reference_candidates(avail, date(2023, 4, 4), lookback_days=30)
    assert out == [date(2023, 3, 20), date(2023, 3, 6)]  # excludes the target itself and later


def test_select_target_dates() -> None:
    avail = [date(2023, 4, 1), date(2023, 4, 4), date(2023, 4, 9)]
    wanted = {date(2023, 4, 3), date(2023, 4, 4), date(2023, 4, 5)}
    assert s2.select_dates(avail, wanted) == [date(2023, 4, 4)]
