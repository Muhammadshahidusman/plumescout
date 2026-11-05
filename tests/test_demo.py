"""Tests for demo verdict text (neutral wording, hard rules 7 and 8)."""

import pytest

from plumescout import demo

BANNED = ["kg", "emission rate", "polluter", "worst", "leak", "violat", "culprit", "blame"]


@pytest.mark.parametrize(
    ("n_det", "statuses", "ref", "emit_tn", "expected"),
    [
        (1, ["tp"], "Carbon Mapper (Tanager)", False, "Plume-like signal detected"),
        (0, [], "EMIT", False, "No plume-like signal detected"),
        (2, ["fp", "fp"], None, True, "EMIT observed this site within 1 day and reported no plume"),
        (1, ["unverified"], None, False, "No reference observation within 1 day"),
        (0, [], None, True, "EMIT observed this site within 1 day and reported no plume"),
    ],
)
def test_verdict_contents(n_det, statuses, ref, emit_tn, expected) -> None:
    v = demo.verdict(n_det, statuses, ref, emit_tn, clear=True, in_train_overlap=False)
    assert expected in v
    assert all(b not in v.lower() for b in BANNED)


def test_verdict_reference_wording() -> None:
    v = demo.verdict(1, ["tp"], "EMIT", False, clear=True, in_train_overlap=False)
    assert "EMIT reported a plume within 1 km and 1 day" in v
    v = demo.verdict(0, [], "EMIT", False, clear=True, in_train_overlap=False)
    assert "reported a plume" in v and "not detected" in v


def test_verdict_cloudy_and_overlap_notes() -> None:
    v = demo.verdict(0, [], None, False, clear=False, in_train_overlap=True)
    assert "cloud" in v.lower() and "excluded from the evaluation" in v
