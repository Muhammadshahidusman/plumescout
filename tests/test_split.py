"""Split tests (hard rule 3: split by site, never by image).

Any change to data splits must keep these passing.
"""

import pandas as pd
import pytest

from plumescout import split
from plumescout.sites import cluster_points

# Real CH4Net coordinates (Vaughan et al. 2024, Table 2) for a few close and far sites.
SITES = pd.DataFrame(
    {
        "site_id": ["T5", "T6", "T7", "T12", "T15", "T16", "T17", "T22", "T23"],
        "lon": [
            53.78836,
            53.77502,
            53.77921,
            54.23498,
            54.20393,
            54.19769,
            54.19764,
            53.92431,
            53.92702,
        ],
        "lat": [
            39.46428,
            39.4616,
            39.45965,
            38.85515,
            38.51871,
            38.50798,
            38.49393,
            37.92913,
            37.71665,
        ],
    }
)


def _images(n_per_site: int = 4) -> pd.DataFrame:
    rows = []
    for sid in SITES.site_id:
        for folder in ("train", "val", "test"):
            for j in range(n_per_site):
                rows.append({"site_id": sid, "folder": folder, "image_id": f"{folder}-{sid}-{j}"})
    return pd.DataFrame(rows)


def test_cluster_points_uses_metres() -> None:
    g = cluster_points(SITES.lon.to_numpy(), SITES.lat.to_numpy(), eps_m=2000)
    by = dict(zip(SITES.site_id, g, strict=True))
    assert by["T5"] == by["T6"] == by["T7"]  # within ~1.2 km
    assert by["T15"] == by["T16"] == by["T17"]  # chained, each under 2 km apart
    assert by["T22"] != by["T23"]  # ~24 km apart
    assert by["T12"] != by["T15"]


def test_site_groups_maps_every_site() -> None:
    groups = split.site_groups(SITES, eps_m=2000)
    assert set(groups) == set(SITES.site_id)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_site_year_split_is_site_disjoint(seed: int) -> None:
    groups = split.site_groups(SITES, eps_m=2000)
    s = split.site_year_split(_images(), groups, test_frac=0.3, val_frac=0.2, seed=seed)
    tr, va, te = (set(s.loc[s.split == k, "site_id"]) for k in ("train", "val", "test"))
    assert tr and va and te
    assert not (tr & te) and not (tr & va) and not (va & te)


@pytest.mark.parametrize("seed", [0, 5])
def test_site_year_split_keeps_groups_together(seed: int) -> None:
    groups = split.site_groups(SITES, eps_m=2000)
    s = split.site_year_split(_images(), groups, test_frac=0.3, val_frac=0.2, seed=seed)
    s = s[s.split != "unused"]
    per_group = s.assign(g=s.site_id.map(groups)).groupby("g").split.nunique()
    assert (per_group == 1).all()


def test_site_year_split_respects_years() -> None:
    """Test rows come only from the 2021 folder; train and val only from 2017 to 2020."""
    groups = split.site_groups(SITES, eps_m=2000)
    s = split.site_year_split(_images(), groups, test_frac=0.3, val_frac=0.2, seed=0)
    assert set(s.loc[s.split == "test", "folder"]) == {"test"}
    assert set(s.loc[s.split.isin(["train", "val"]), "folder"]) <= {"train", "val"}


def test_site_year_split_is_deterministic() -> None:
    groups = split.site_groups(SITES, eps_m=2000)
    a = split.site_year_split(_images(), groups, test_frac=0.3, val_frac=0.2, seed=7)
    b = split.site_year_split(_images(), groups, test_frac=0.3, val_frac=0.2, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_assert_site_disjoint_raises_on_leak() -> None:
    with pytest.raises(AssertionError):
        split.assert_site_disjoint({"T1", "T2"}, {"T2", "T3"})


def test_paper_split_is_flagged_reproduction_only() -> None:
    s = split.paper_split(_images())
    assert s.attrs["reproduction_only"] is True
    assert set(s.split) == {"train", "val", "test"}
