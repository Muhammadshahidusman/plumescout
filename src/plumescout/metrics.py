"""Detection metrics on binary masks (same 2D pixel grid, no CRS involved).

Plume level: a labelled plume (connected component) is detected if any predicted
component overlaps it; a predicted component is matched if it overlaps any labelled
plume. Components smaller than `min_px` are dropped from both. Scene level: a scene
is flagged if it has at least one predicted component; positive if it has a label.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage

_EIGHT = np.ones((3, 3), dtype=bool)


def components(mask: np.ndarray, min_px: int) -> list[np.ndarray]:
    """8-connected components of `mask` with at least `min_px` pixels, as boolean masks."""
    lab, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    return [lab == i for i in range(1, n + 1) if sizes[i] >= min_px]


def plume_counts(
    pred: np.ndarray, label: np.ndarray, min_px: int, valid: np.ndarray | None = None
) -> dict[str, int]:
    """Object counts for one scene. Optional `valid` restricts both masks (e.g. drops padding)."""
    pred = np.asarray(pred, dtype=bool)
    label = np.asarray(label, dtype=bool)
    if valid is not None:
        pred, label = pred & valid, label & valid
    pc, lc = components(pred, min_px), components(label, min_px)
    pred_any = np.any(pc, axis=0) if pc else np.zeros_like(pred)
    lab_any = np.any(lc, axis=0) if lc else np.zeros_like(label)
    return {
        "label_plumes": len(lc),
        "detected_plumes": sum(bool((c & pred_any).any()) for c in lc),
        "pred_objects": len(pc),
        "pred_matched": sum(bool((c & lab_any).any()) for c in pc),
    }


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def summarise(rows: Iterable[dict[str, int]]) -> dict[str, float]:
    """Pool per-scene counts into plume- and scene-level recall, precision and F1."""
    rows = list(rows)
    lp = sum(r["label_plumes"] for r in rows)
    dp = sum(r["detected_plumes"] for r in rows)
    po = sum(r["pred_objects"] for r in rows)
    pm = sum(r["pred_matched"] for r in rows)
    pos = [r["label_plumes"] > 0 for r in rows]
    flag = [r["pred_objects"] > 0 for r in rows]
    tp_s = sum(p and f for p, f in zip(pos, flag, strict=True))
    out = {
        "n_scenes": len(rows),
        "n_positive_scenes": sum(pos),
        "n_label_plumes": lp,
        "n_pred_objects": po,
        "plume_recall": _ratio(dp, lp),
        "plume_precision": _ratio(pm, po),
        "scene_recall": _ratio(tp_s, sum(pos)),
        "scene_precision": _ratio(tp_s, sum(flag)),
    }
    for lvl in ("plume", "scene"):
        r, p = out[f"{lvl}_recall"], out[f"{lvl}_precision"]
        out[f"{lvl}_f1"] = _ratio(2 * r * p, r + p) if r == r and p == p else float("nan")
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel intersection over union of two boolean masks (nan if both empty)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return _ratio((a & b).sum(), (a | b).sum())


def best_threshold(scores: dict[float, dict[str, float]], key: str = "plume_f1") -> float:
    """Threshold with the highest `key` score (nan counts as 0; ties go to the lowest)."""
    return max(sorted(scores), key=lambda t: np.nan_to_num(scores[t][key]))


def scene_flag(prob: np.ndarray, thr: float, min_px: int, valid: np.ndarray | None = None) -> bool:
    """CH4Net paper scene rule: True if any 8-connected region with prob >= thr has >= min_px px.

    The paper says "greater than 115 pixels", i.e. min_px = 116.
    """
    m = np.asarray(prob) >= thr
    if valid is not None:
        m &= valid
    return len(components(m, min_px)) > 0


def scene_scores(positive: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """Scene-level recall, precision, false positive rate, accuracy and balanced accuracy."""
    pos, flag = np.asarray(positive, bool), np.asarray(flagged, bool)
    tp, fp = int((pos & flag).sum()), int((~pos & flag).sum())
    rec, fpr = _ratio(tp, pos.sum()), _ratio(fp, (~pos).sum())
    return {
        "n_scenes": int(pos.size),
        "n_positive_scenes": int(pos.sum()),
        "recall": rec,
        "precision": _ratio(tp, flag.sum()),
        "fpr": fpr,
        "accuracy": float((pos == flag).mean()) if pos.size else float("nan"),
        "balanced_accuracy": (rec + 1 - fpr) / 2,
    }


def bootstrap_scene_ci(
    positive: np.ndarray,
    flagged: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI for recall, precision, fpr and balanced accuracy.

    Without `groups`, resamples scenes; scenes are not independent within a site, so that
    CI is optimistic. With `groups` (e.g. site ids), resamples whole groups (cluster
    bootstrap), the honest choice when there are few sites.
    """
    pos, flag = np.asarray(positive, bool), np.asarray(flagged, bool)
    rng = np.random.default_rng(seed)
    keys = ("recall", "precision", "fpr", "balanced_accuracy")
    draws = {k: [] for k in keys}
    if groups is not None:
        g = np.asarray(groups)
        members = [np.flatnonzero(g == u) for u in np.unique(g)]
    for _ in range(n_boot):
        if groups is None:
            i = rng.integers(0, pos.size, pos.size)
        else:
            i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        s = scene_scores(pos[i], flag[i])
        for k in keys:
            draws[k].append(s[k])
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {k: tuple(float(v) for v in np.nanpercentile(draws[k], q)) for k in keys}
