"""Pool the 5 site-disjoint CV folds (each 2021 scene is test in exactly one fold).

Reads models/cv5_fold{k}/test_scenes.parquet and metrics.json; writes models/cv5_summary.json.
Scene rule: CH4Net paper (region >= 116 px at p >= thr). CIs: site bootstrap over all sites.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from plumescout import metrics
from plumescout.config import resolve_path

logger = logging.getLogger(__name__)


def pool(frames: list[pd.DataFrame], flag_col: str) -> dict:
    """Pooled scene scores and site-bootstrap CI for one flag column per fold."""
    d = pd.concat(frames, ignore_index=True)
    pos, flag = d.has_plume.to_numpy(bool), d[flag_col].to_numpy(bool)
    out = metrics.scene_scores(pos, flag)
    out["n_sites"] = int(d.site_id.nunique())
    out["ci95_site_bootstrap"] = metrics.bootstrap_scene_ci(
        pos, flag, seed=42, groups=d.site_id.to_numpy()
    )
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    frames_val, frames_025, folds = [], [], []
    for k in range(5):
        run = resolve_path(f"models/cv5_fold{k}")
        ts = pd.read_parquet(run / "test_scenes.parquet")
        m = json.loads((run / "metrics.json").read_text())
        thr = float(ts.scene_thr_val.iloc[0])
        frames_val.append(ts.assign(flag=ts[f"scene_flag_{thr}"]))
        frames_025.append(ts.assign(flag=ts["scene_flag_0.25"]))
        s = metrics.scene_scores(
            ts.has_plume.to_numpy(bool), ts[f"scene_flag_{thr}"].to_numpy(bool)
        )
        folds.append(
            {
                "fold": k,
                "best_epoch": m["best_epoch"],
                "thr_val": thr,
                "test_sites": m["test_sites"],
                **{
                    x: s[x]
                    for x in (
                        "n_scenes",
                        "n_positive_scenes",
                        "recall",
                        "precision",
                        "fpr",
                        "balanced_accuracy",
                    )
                },
            }
        )
    out = {
        "pooled_thr_from_val": pool(frames_val, "flag"),
        "pooled_thr_0.25": pool(frames_025, "flag"),
        "folds": folds,
        "note": "Each fold picks its threshold and checkpoint on its own val groups.",
    }
    path = resolve_path("models/cv5_summary.json")
    path.write_text(json.dumps(out, indent=2, default=float))
    for k in ("pooled_thr_from_val", "pooled_thr_0.25"):
        p = out[k]
        logger.info(
            "%s: %d scenes, %d positive, %d sites, recall %.3f, precision %.3f, FPR %.3f, "
            "bal. acc %.3f %s",
            k,
            p["n_scenes"],
            p["n_positive_scenes"],
            p["n_sites"],
            p["recall"],
            p["precision"],
            p["fpr"],
            p["balanced_accuracy"],
            np.round(p["ci95_site_bootstrap"]["balanced_accuracy"], 3),
        )


if __name__ == "__main__":
    main()
