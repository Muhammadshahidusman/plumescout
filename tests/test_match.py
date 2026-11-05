"""Tests for plumescout.match (TRD 5.7): TP / FP / FN / unverified."""

import pandas as pd

from plumescout import match

T0 = pd.Timestamp("2023-04-05T07:30Z")  # S2 overpass
LAT0, LON0 = 29.0, 47.9
DLON_1KM = 1000 / 97250  # degrees longitude per km at 29 N


def _dets(rows):
    return pd.DataFrame(
        rows, columns=["det_id", "scene_id", "site_id", "datetime_utc", "lon", "lat"]
    )


def _refs(rows):
    return pd.DataFrame(rows, columns=["ref_id", "source", "site_id", "datetime_utc", "lon", "lat"])


def _obs(rows):
    return pd.DataFrame(rows, columns=["site_id", "datetime_utc", "emit_plume"])


def _scenes(rows):
    return pd.DataFrame(rows, columns=["scene_id", "site_id", "datetime_utc", "clear"])


def test_tp_within_1km_and_1day() -> None:
    d = _dets([["d1", "s1", "A", T0, LON0 + 0.8 * DLON_1KM, LAT0]])
    r = _refs([["e1", "emit", "A", T0 - pd.Timedelta(hours=2), LON0, LAT0]])
    out = match.classify_detections(d, r, _obs([]), 1000, pd.Timedelta(days=1))
    assert out.loc[0, "match_status"] == "tp" and out.loc[0, "match_source"] == "emit"
    assert out.loc[0, "match_ref_id"] == "e1"


def test_too_far_in_space_or_time_is_not_tp() -> None:
    d = _dets(
        [
            ["far", "s1", "A", T0, LON0 + 1.2 * DLON_1KM, LAT0],
            ["late", "s2", "A", T0 + pd.Timedelta(days=1, hours=1), LON0, LAT0],
        ]
    )
    r = _refs([["e1", "emit", "A", T0, LON0, LAT0]])
    out = match.classify_detections(d, r, _obs([]), 1000, pd.Timedelta(days=1))
    assert set(out.match_status) == {"unverified"}


def test_fp_needs_emit_true_negative_coverage() -> None:
    d = _dets(
        [
            ["covered", "s1", "A", T0, LON0, LAT0],
            ["uncovered", "s2", "B", T0, LON0, LAT0],
        ]
    )
    obs = _obs([["A", T0 - pd.Timedelta(hours=20), False], ["B", T0, True]])
    out = match.classify_detections(d, _refs([]), obs, 1000, pd.Timedelta(days=1)).set_index(
        "det_id"
    )
    assert out.loc["covered", "match_status"] == "fp"
    # B: EMIT saw a plume, but no plume complex reference is near: not a true negative
    assert out.loc["uncovered", "match_status"] == "unverified"


def test_reference_beats_true_negative() -> None:
    # Carbon Mapper plume the same day as an EMIT no-plume overpass: detection is a TP
    d = _dets([["d1", "s1", "A", T0, LON0, LAT0]])
    r = _refs([["c1", "cm", "A", T0 + pd.Timedelta(hours=1), LON0, LAT0]])
    obs = _obs([["A", T0, False]])
    out = match.classify_detections(d, r, obs, 1000, pd.Timedelta(days=1))
    assert out.loc[0, "match_status"] == "tp" and out.loc[0, "match_source"] == "cm"


def test_false_negatives_only_on_clear_scenes() -> None:
    r = _refs(
        [
            ["hit", "emit", "A", T0, LON0, LAT0],
            ["miss", "cm", "A", T0 + pd.Timedelta(days=10), LON0, LAT0],
            ["cloudy", "cm", "A", T0 + pd.Timedelta(days=20), LON0, LAT0],
            ["no_s2", "cm", "A", T0 + pd.Timedelta(days=40), LON0, LAT0],
        ]
    )
    sc = _scenes(
        [
            ["s1", "A", T0, True],
            ["s2", "A", T0 + pd.Timedelta(days=10), True],
            ["s3", "A", T0 + pd.Timedelta(days=20), False],
        ]
    )
    d = _dets([["d1", "s1", "A", T0, LON0, LAT0]])
    out = match.reference_outcomes(r, d, sc, 1000, pd.Timedelta(days=1)).set_index("ref_id")
    assert out.loc["hit", "outcome"] == "detected"
    assert out.loc["miss", "outcome"] == "fn"
    assert out.loc["cloudy", "outcome"] == "no_clear_scene"
    assert out.loc["no_s2", "outcome"] == "no_clear_scene"


def test_detection_on_cloudy_scene_still_counts_as_detected() -> None:
    r = _refs([["e1", "emit", "A", T0, LON0, LAT0]])
    sc = _scenes([["s1", "A", T0, False]])
    d = _dets([["d1", "s1", "A", T0, LON0, LAT0]])
    out = match.reference_outcomes(r, d, sc, 1000, pd.Timedelta(days=1))
    assert out.loc[0, "outcome"] == "detected"


def test_detections_from_other_sites_never_match() -> None:
    d = _dets([["d1", "s1", "B", T0, LON0, LAT0]])
    r = _refs([["e1", "emit", "A", T0, LON0, LAT0]])
    out = match.classify_detections(d, r, _obs([]), 1000, pd.Timedelta(days=1))
    assert out.loc[0, "match_status"] == "unverified"
