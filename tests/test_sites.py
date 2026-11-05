"""Tests for Gulf site building, train-overlap flag and EMIT true negatives (TRD 5.1, 5.7)."""

import numpy as np
import pandas as pd
import pytest

from plumescout import sites as st

# 0.01 deg latitude = 1,111.9 m; at 29 N, 0.01 deg longitude = 972.5 m
LAT0, LON0 = 29.0, 47.9


def test_haversine_known_distances() -> None:
    assert st.haversine_m(LON0, LAT0, LON0, LAT0 + 0.01) == pytest.approx(1111.95, rel=1e-3)
    assert st.haversine_m(LON0, LAT0, LON0 + 0.01, LAT0) == pytest.approx(972.5, rel=2e-3)


def test_nearest_distance() -> None:
    d, i = st.nearest_m(
        np.array([LON0]),
        np.array([LAT0]),
        np.array([LON0 + 1, LON0]),
        np.array([LAT0, LAT0 + 0.02]),
    )
    assert i[0] == 1 and d[0] == pytest.approx(2223.9, rel=1e-3)


def _refs() -> pd.DataFrame:
    t = pd.Timestamp("2023-04-05T06:35:00Z")
    return pd.DataFrame(
        {
            "ref_id": ["e1", "e2", "c1", "c2", "o1"],
            "source": ["emit", "emit", "cm", "cm", "emit"],
            "instrument": ["EMIT", "EMIT", "Tanager", "Tanager", "EMIT"],
            "datetime_utc": [t, t + pd.Timedelta(days=30), t, t, t],
            # e1, e2, c1 within ~0.5 km; c2 3 km east; o1 outside every AOI
            "lon": [LON0, LON0 + 0.004, LON0 + 0.002, LON0 + 0.03, 10.0],
            "lat": [LAT0, LAT0, LAT0 + 0.002, LAT0, 10.0],
        }
    )


AOIS = {"kuwait_burgan": {"region": "Kuwait", "bbox": [47.7, 28.8, 48.2, 29.2], "role": "test"}}


def test_build_sites_clusters_within_1km_and_drops_outside_aoi() -> None:
    sites, refs = st.build_sites(_refs(), AOIS, eps_m=1000, train=pd.DataFrame(), overlap_m=2000)
    assert len(sites) == 2
    assert "o1" not in set(refs.ref_id)
    big = sites.loc[sites.n_refs.idxmax()]
    assert big.n_emit == 2 and big.n_cm == 1 and big.region == "Kuwait"
    assert refs.set_index("ref_id").loc[["e1", "e2", "c1"], "site_id"].nunique() == 1
    assert sites.site_id.str.startswith("KWT").all()


def test_train_overlap_flag_at_2km() -> None:
    train = pd.DataFrame(
        {"lon": [LON0 + 0.03 + 0.015], "lat": [LAT0], "train_source": ["MARS-S2L"]}
    )  # 1.46 km from c2, ~4.3 km from the e1 cluster
    sites, _ = st.build_sites(_refs(), AOIS, eps_m=1000, train=train, overlap_m=2000)
    s = sites.set_index("n_refs")
    assert bool(s.loc[1, "in_train_overlap"]) and s.loc[1, "nearest_train_source"] == "MARS-S2L"
    assert not bool(s.loc[3, "in_train_overlap"])
    assert s.loc[3, "nearest_train_m"] > 2000


def test_emit_overpasses_true_negative_and_plume() -> None:
    sites = pd.DataFrame({"site_id": ["A", "B"], "lon": [LON0, LON0 + 0.5], "lat": [LAT0, LAT0]})
    t1, t2 = pd.Timestamp("2023-04-05T06:35:00Z"), pd.Timestamp("2023-05-01T07:00:00Z")
    box = f"POLYGON(({LON0 - 0.1} {LAT0 - 0.1}, {LON0 + 0.1} {LAT0 - 0.1}, {LON0 + 0.1} " + (
        f"{LAT0 + 0.1}, {LON0 - 0.1} {LAT0 + 0.1}, {LON0 - 0.1} {LAT0 - 0.1}))"
    )
    over = pd.DataFrame(
        {"granule_id": ["g1", "g2"], "datetime_utc": [t1, t2], "footprint_wkt": [box, box]}
    )
    plumes = pd.DataFrame(
        {"datetime_utc": [t1 + pd.Timedelta(seconds=20)], "lon": [LON0 + 0.003], "lat": [LAT0]}
    )
    obs = st.emit_observations(sites, over, plumes, radius_m=1000, max_dt=pd.Timedelta(hours=1))
    assert set(obs.site_id) == {"A"}  # B is outside both footprints
    o = obs.set_index("granule_id")
    assert bool(o.loc["g1", "emit_plume"]) and not bool(o.loc["g2", "emit_plume"])


def test_target_dates_unique_and_padded() -> None:
    t = pd.to_datetime(["2023-04-05T06:35Z", "2023-04-05T23:50Z", "2023-04-06T01:00Z"], utc=True)
    days = st.target_dates(pd.Series(t), pad_days=1)
    assert days == sorted(
        {pd.Timestamp(d).date() for d in pd.date_range("2023-04-04", "2023-04-07")}
    )


def test_pair_with_scenes_within_one_day_utc() -> None:
    events = pd.DataFrame(
        {
            "site_id": ["A", "A", "B"],
            "datetime_utc": pd.to_datetime(
                ["2023-04-05T06:35Z", "2023-05-01T06:00Z", "2023-04-05T06:35Z"], utc=True
            ),
        }
    )
    scenes = pd.DataFrame(
        {
            "site_id": ["A", "A", "A", "B"],
            "datetime_utc": pd.to_datetime(
                [
                    "2023-04-06T07:30Z",
                    "2023-04-03T07:30Z",
                    "2023-05-02T05:59Z",
                    "2023-04-05T07:30Z",
                ],
                utc=True,
            ),
            "scene_id": ["A1", "A2", "A3", "B1"],
            "clear": [True, True, False, True],
        }
    )
    p = st.pair_with_scenes(events, scenes, max_dt=pd.Timedelta(days=1))
    # A event 1: A1 is 24 h 55 min later -> outside +-1 day; A2 2 days earlier -> outside
    assert p.loc[0, "n_scenes"] == 0
    # A event 2: A3 23 h 59 min later -> inside, but not clear
    assert p.loc[1, "n_scenes"] == 1 and p.loc[1, "n_clear"] == 0
    # B event: B1 55 min later; scenes of other sites never pair
    assert p.loc[2, "n_clear"] == 1 and p.loc[2, "clear_scene_ids"] == "B1"


def test_site_table_counts_clear_pairs() -> None:
    t0 = pd.Timestamp("2023-04-05T06:35Z")
    sites = pd.DataFrame({"site_id": ["A"], "region": ["Kuwait"], "in_train_overlap": [False]})
    refs = pd.DataFrame(
        {
            "site_id": ["A", "A"],
            "source": ["emit", "cm"],
            "instrument": ["EMIT", "Tanager"],
            "datetime_utc": [t0, t0 + pd.Timedelta(days=10)],
        }
    )
    obs = pd.DataFrame(
        {
            "site_id": ["A", "A"],
            "datetime_utc": [t0, t0 + pd.Timedelta(days=20)],
            "emit_plume": [True, False],
        }
    )
    scenes = pd.DataFrame(
        {
            "site_id": ["A"] * 3,
            "scene_id": ["s1", "s2", "s3"],
            "datetime_utc": [
                t0 + pd.Timedelta(hours=1),
                t0 + pd.Timedelta(days=10, hours=2),
                t0 + pd.Timedelta(days=20, hours=3),
            ],
            "obscured_frac": [0.0, 0.5, 0.1],
        }
    )
    t, _ = st.site_table(sites, refs, obs, scenes, pd.Timedelta(days=1), 0.2)
    r = t.set_index("site_id").loc["A"]
    assert r.emit_pairs_clear_plume == 1 and r.emit_pairs_clear_tn == 1
    assert r.cm_plumes == 1 and r.cm_pairs_clear == 0  # its only scene is cloudy
    assert t.iloc[-1].site_id == "TOTAL"
