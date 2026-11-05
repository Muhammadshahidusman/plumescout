"""Tests for plumescout.metrics (plume-level and scene-level detection scores)."""

import numpy as np

from plumescout import metrics


def _blob(shape, r0, r1, c0, c1):
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_components_drop_small() -> None:
    m = _blob((20, 20), 0, 5, 0, 5) | _blob((20, 20), 10, 12, 10, 12)  # 25 px and 4 px
    comps = metrics.components(m, min_px=10)
    assert len(comps) == 1 and comps[0].sum() == 25


def test_plume_counts_tp_fn_fp() -> None:
    shape = (40, 40)
    label = _blob(shape, 0, 6, 0, 6) | _blob(shape, 20, 26, 20, 26)  # 2 plumes
    pred = _blob(shape, 2, 8, 2, 8) | _blob(shape, 30, 36, 0, 6)  # hits plume 1, one FP
    c = metrics.plume_counts(pred, label, min_px=10)
    assert c == {"label_plumes": 2, "detected_plumes": 1, "pred_objects": 2, "pred_matched": 1}


def test_plume_counts_empty() -> None:
    z = np.zeros((10, 10), dtype=bool)
    assert metrics.plume_counts(z, z, min_px=10) == {
        "label_plumes": 0,
        "detected_plumes": 0,
        "pred_objects": 0,
        "pred_matched": 0,
    }


def test_valid_mask_excludes_padding() -> None:
    shape = (20, 20)
    pred = _blob(shape, 15, 20, 15, 20)  # 25 px, entirely in padding
    valid = _blob(shape, 0, 12, 0, 12)
    c = metrics.plume_counts(pred, np.zeros(shape, bool), min_px=10, valid=valid)
    assert c["pred_objects"] == 0


def test_summarise_rates() -> None:
    rows = [
        {"label_plumes": 2, "detected_plumes": 1, "pred_objects": 2, "pred_matched": 1},
        {"label_plumes": 0, "detected_plumes": 0, "pred_objects": 1, "pred_matched": 0},
        {"label_plumes": 1, "detected_plumes": 1, "pred_objects": 1, "pred_matched": 1},
    ]
    s = metrics.summarise(rows)
    assert s["plume_recall"] == 2 / 3
    assert s["plume_precision"] == 2 / 4
    assert s["scene_recall"] == 2 / 2  # both positive scenes have a detection
    assert s["scene_precision"] == 2 / 3  # 3 scenes flagged, 2 truly positive
    assert s["n_scenes"] == 3 and s["n_label_plumes"] == 3


def test_iou() -> None:
    a = _blob((10, 10), 0, 4, 0, 4)
    b = _blob((10, 10), 2, 6, 2, 6)
    assert metrics.iou(a, b) == 4 / 28


def test_best_threshold_picks_max_f1_and_ignores_nan() -> None:
    from plumescout.metrics import best_threshold

    scores = {
        0.3: {"plume_f1": 0.40},
        0.5: {"plume_f1": 0.55},
        0.7: {"plume_f1": float("nan")},
        0.9: {"plume_f1": 0.10},
    }
    assert best_threshold(scores) == 0.5


def test_best_threshold_ties_go_to_lowest_threshold() -> None:
    from plumescout.metrics import best_threshold

    assert best_threshold({0.6: {"plume_f1": 0.5}, 0.4: {"plume_f1": 0.5}}) == 0.4


def test_scene_rule_needs_component_above_min_px() -> None:
    from plumescout.metrics import scene_flag

    p = np.zeros((20, 20))
    p[:10, :10] = 0.3  # 100 px component at p = 0.3
    assert scene_flag(p, thr=0.25, min_px=100)
    assert not scene_flag(p, thr=0.25, min_px=101)
    assert not scene_flag(p, thr=0.35, min_px=10)


def test_scene_scores_predict_everything_gives_half_balanced_accuracy() -> None:
    from plumescout.metrics import scene_scores

    pos = np.array([True, False, False, True])
    s = scene_scores(pos, np.ones(4, bool))
    assert s["recall"] == 1.0 and s["fpr"] == 1.0 and s["balanced_accuracy"] == 0.5
    s = scene_scores(pos, np.array([True, False, False, False]))
    assert s["recall"] == 0.5 and s["fpr"] == 0.0 and s["precision"] == 1.0
    assert s["balanced_accuracy"] == 0.75


def test_bootstrap_scene_ci_brackets_point_and_is_deterministic() -> None:
    from plumescout.metrics import bootstrap_scene_ci, scene_scores

    rng = np.random.default_rng(1)
    pos = rng.random(400) < 0.3
    flag = np.where(pos, rng.random(400) < 0.8, rng.random(400) < 0.2)
    ci = bootstrap_scene_ci(pos, flag, n_boot=500, seed=0)
    pt = scene_scores(pos, flag)
    for k in ("recall", "precision", "fpr", "balanced_accuracy"):
        lo, hi = ci[k]
        assert lo <= pt[k] <= hi and hi - lo < 0.3
    assert ci == bootstrap_scene_ci(pos, flag, n_boot=500, seed=0)


def test_bootstrap_scene_ci_zero_width_when_perfect() -> None:
    from plumescout.metrics import bootstrap_scene_ci

    pos = np.array([True, False] * 50)
    assert bootstrap_scene_ci(pos, pos, n_boot=200)["recall"] == (1.0, 1.0)


def test_bootstrap_by_group_is_wider_when_sites_differ() -> None:
    from plumescout.metrics import bootstrap_scene_ci

    # 4 sites: two detected perfectly, two missed entirely
    pos = np.array([True, False] * 40)
    groups = np.repeat(["A", "B", "C", "D"], 20)
    flag = pos & np.isin(groups, ["A", "B"])
    scene = bootstrap_scene_ci(pos, flag, n_boot=500)
    site = bootstrap_scene_ci(pos, flag, n_boot=500, groups=groups)
    w = lambda ci: ci["recall"][1] - ci["recall"][0]  # noqa: E731
    assert w(site) > w(scene)
    assert site["recall"][0] <= 0.25 and site["recall"][1] >= 0.75
