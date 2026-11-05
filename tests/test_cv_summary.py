import pandas as pd
import pytest

from plumescout.cv_summary import pool


def _fold(site: str, has_plume: list[bool], flag: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"site_id": site, "has_plume": has_plume, "flag": flag})


def test_pool_concatenates_folds_and_counts_sites():
    a = _fold("T1", [True, True, False, False], [True, False, False, True])
    b = _fold("T2", [True, False, False], [True, False, False])
    out = pool([a, b], "flag")
    assert out["n_scenes"] == 7
    assert out["n_positive_scenes"] == 3
    assert out["n_sites"] == 2
    assert out["recall"] == pytest.approx(2 / 3)  # 2 of 3 plume scenes flagged
    assert out["fpr"] == pytest.approx(1 / 4)  # 1 of 4 plume-free scenes flagged
    assert out["precision"] == pytest.approx(2 / 3)


def test_pool_ci_is_site_bootstrap_and_brackets_estimate():
    frames = [_fold(f"T{i}", [True, False] * 5, [i % 2 == 0, False] * 5) for i in range(6)]
    out = pool(frames, "flag")
    lo, hi = out["ci95_site_bootstrap"]["recall"]
    assert lo <= out["recall"] <= hi
    # three sites detect every plume, three detect none: site resampling must show that spread
    assert lo < 0.5 < hi
