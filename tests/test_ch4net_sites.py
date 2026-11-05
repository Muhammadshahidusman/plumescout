"""Tests for plumescout.ch4net_sites (site ids for all CH4Net images)."""

import numpy as np
import pytest

from plumescout import ch4net_sites as cs


def test_shape_runs_splits_on_change() -> None:
    shapes = ["a", "a", "b", "b", "b", "a"]
    assert cs.shape_runs(shapes) == [(0, 2), (2, 5), (5, 6)]


def test_dedupe_consecutive() -> None:
    assert cs.dedupe_consecutive(["T5", "T5", "T1", "T1", "T2"]) == ["T5", "T1", "T2"]


def test_segment_ordered_recovers_boundaries() -> None:
    rng = np.random.default_rng(0)
    truth = [0] * 30 + [1] * 50 + [2] * 20
    scores = rng.normal(0, 0.3, (100, 3))
    scores[np.arange(100), truth] += 1.0
    scores[5, 2] += 5.0  # one outlier image must not break the order
    seg = cs.segment_ordered(scores)
    assert seg.tolist() == truth


def test_segment_ordered_single_site() -> None:
    assert cs.segment_ordered(np.zeros((7, 1))).tolist() == [0] * 7


def test_segment_ordered_needs_enough_images() -> None:
    with pytest.raises(ValueError):
        cs.segment_ordered(np.zeros((2, 3)))


def test_assign_blocks_respects_capacity() -> None:
    # 4 blocks, sites A (2 blocks) and B, C (1 each); B is slightly better for block 1
    s = np.array(
        [
            [0.9, 0.1, 0.1],
            [0.8, 0.85, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.9],
        ]
    )
    out = cs.assign_blocks(s, ["A", "B", "C"], {"A": 2, "B": 1, "C": 1})
    assert out == ["A", "A", "B", "C"]


def test_assign_blocks_capacity_must_match_blocks() -> None:
    with pytest.raises(ValueError):
        cs.assign_blocks(np.zeros((3, 2)), ["A", "B"], {"A": 1, "B": 1})


def test_label_order_violations() -> None:
    # within a site CH4Net lists negatives first, then plume images
    assert cs.label_order_violations([0, 0, 0, 1, 1]) == 0
    assert cs.label_order_violations([0, 1, 0, 1]) == 1
    assert cs.label_order_violations([1, 0, 0]) == 2
