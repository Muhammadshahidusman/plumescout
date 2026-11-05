"""Tests for plumescout.evaluate (Gulf metrics, site bootstrap)."""

import numpy as np
import pandas as pd
import pytest

from plumescout import evaluate as ev


def _per_site():
    return pd.DataFrame(
        {
            "site_id": ["A", "B", "C"],
            "detected": [3, 0, 1],
            "fn": [1, 2, 1],
            "tp_dets": [3, 0, 1],
            "fp_dets": [1, 4, 0],
            "clear_tn_scenes": [10, 20, 10],
        }
    )


def test_pooled_metrics() -> None:
    m = ev.pooled(_per_site())
    assert m["plume_recall"] == pytest.approx(4 / 8)
    assert m["precision"] == pytest.approx(4 / 9)
    assert m["fp_per_100_clear_tn"] == pytest.approx(5 / 40 * 100)
    assert m["n_sites"] == 3 and m["n_ref_plumes"] == 8


def test_pooled_nan_when_empty() -> None:
    m = ev.pooled(_per_site().assign(detected=0, fn=0, tp_dets=0, fp_dets=0))
    assert np.isnan(m["plume_recall"]) and np.isnan(m["precision"])


def test_site_bootstrap_brackets_point_and_is_deterministic() -> None:
    ps = _per_site()
    ci = ev.site_bootstrap(ps, n_boot=500, seed=1)
    pt = ev.pooled(ps)
    for k in ("plume_recall", "precision", "fp_per_100_clear_tn"):
        lo, hi = ci[k]
        assert lo <= pt[k] <= hi
    assert ci == ev.site_bootstrap(ps, n_boot=500, seed=1)


def test_per_site_counts() -> None:
    dets = pd.DataFrame(
        {
            "site_id": ["A", "A", "A", "B"],
            "method": ["unet"] * 4,
            "match_status": ["tp", "fp", "unverified", "fp"],
            "flare_masked": [False, False, False, True],
        }
    )
    refs = pd.DataFrame(
        {"site_id": ["A", "A", "B"], "outcome": ["detected", "fn", "no_clear_scene"]}
    )
    tn = pd.DataFrame({"site_id": ["A", "B", "B"]})
    ps = ev.per_site_counts(dets, refs, tn, ["A", "B"])
    a, b = ps.set_index("site_id").loc["A"], ps.set_index("site_id").loc["B"]
    assert (a.detected, a.fn, a.tp_dets, a.fp_dets, a.unverified, a.clear_tn_scenes) == (
        1,
        1,
        1,
        1,
        1,
        1,
    )
    # flare-masked detections are removed before counting
    assert (b.fp_dets, b.flare_masked, b.no_clear_scene, b.clear_tn_scenes) == (0, 1, 1, 2)


def test_fp_cause_order() -> None:
    assert ev.fp_cause(fire_within_2km=True, cloud_or_shadow=0.5, b12_rank=0.99) == "flare_nearby"
    assert ev.fp_cause(fire_within_2km=False, cloud_or_shadow=0.02, b12_rank=0.99) == "cloud_edge"
    assert (
        ev.fp_cause(fire_within_2km=False, cloud_or_shadow=0.0, b12_rank=0.95) == "bright_surface"
    )
    assert ev.fp_cause(fire_within_2km=False, cloud_or_shadow=0.0, b12_rank=0.5) == "unknown"
