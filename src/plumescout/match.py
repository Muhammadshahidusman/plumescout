"""Match S2 detections to EMIT V002 and Carbon Mapper references (TRD 5.7).

Rules (distances in metres via haversine, times as UTC datetimes):
- Detection `tp`: a reference plume at the same site within max_m and |dt| <= max_dt.
- Detection `fp`: no such plume, and EMIT observed the site within |dt| <= max_dt and
  reported no plume there (EMIT true negative).
- Detection `unverified`: neither; logged, never counted.
- Reference `detected`: a detection at the same site within max_m and max_dt on any scene.
  `fn`: not detected although a clear S2 scene exists within max_dt.
  `no_clear_scene`: not detected and no clear S2 scene within max_dt (not counted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plumescout.sites import haversine_m


def _near(
    lon: float, lat: float, t: pd.Timestamp, cand: pd.DataFrame, max_m: float, max_dt
) -> pd.DataFrame:
    """Rows of cand within max_m and |dt| <= max_dt, with a `dist_m` column, nearest first."""
    if cand.empty:
        return cand.assign(dist_m=pd.Series(dtype=float))
    c = cand[(cand.datetime_utc - t).abs() <= max_dt]
    if c.empty:
        return c.assign(dist_m=pd.Series(dtype=float))
    c = c.assign(dist_m=haversine_m(lon, lat, c.lon.to_numpy(), c.lat.to_numpy()))
    return c[c.dist_m <= max_m].sort_values("dist_m")


def classify_detections(
    dets: pd.DataFrame,
    refs: pd.DataFrame,
    emit_obs: pd.DataFrame,
    max_m: float,
    max_dt: pd.Timedelta,
) -> pd.DataFrame:
    """Add match_status (tp / fp / unverified), match_source and match_ref_id to detections.

    dets: det_id, site_id, datetime_utc, lon, lat. refs: ref_id, source, site_id,
    datetime_utc, lon, lat. emit_obs: site_id, datetime_utc, emit_plume.
    """
    status, source, ref_id = [], [], []
    refs_by = {k: g for k, g in refs.groupby("site_id")} if len(refs) else {}
    tn = emit_obs[~emit_obs.emit_plume.astype(bool)] if len(emit_obs) else emit_obs
    tn_by = {k: g for k, g in tn.groupby("site_id")} if len(tn) else {}
    for d in dets.itertuples():
        hit = _near(
            d.lon, d.lat, d.datetime_utc, refs_by.get(d.site_id, refs.iloc[0:0]), max_m, max_dt
        )
        if len(hit):
            status.append("tp")
            source.append(hit.iloc[0].source)
            ref_id.append(hit.iloc[0].ref_id)
            continue
        g = tn_by.get(d.site_id)
        covered = g is not None and bool(((g.datetime_utc - d.datetime_utc).abs() <= max_dt).any())
        status.append("fp" if covered else "unverified")
        source.append("emit" if covered else "none")
        ref_id.append(None)
    out = dets.copy()
    out["match_status"], out["match_source"], out["match_ref_id"] = status, source, ref_id
    return out


def reference_outcomes(
    refs: pd.DataFrame,
    dets: pd.DataFrame,
    scenes: pd.DataFrame,
    max_m: float,
    max_dt: pd.Timedelta,
) -> pd.DataFrame:
    """Add outcome (detected / fn / no_clear_scene) and n_clear_scenes to references.

    scenes: scene_id, site_id, datetime_utc, clear.
    """
    dets_by = {k: g for k, g in dets.groupby("site_id")} if len(dets) else {}
    sc_by = {k: g for k, g in scenes.groupby("site_id")} if len(scenes) else {}
    outcome, n_clear = [], []
    for r in refs.itertuples():
        d = dets_by.get(r.site_id)
        found = d is not None and len(_near(r.lon, r.lat, r.datetime_utc, d, max_m, max_dt)) > 0
        s = sc_by.get(r.site_id)
        nc = (
            0
            if s is None
            else int((((s.datetime_utc - r.datetime_utc).abs() <= max_dt) & s.clear).sum())
        )
        n_clear.append(nc)
        outcome.append("detected" if found else ("fn" if nc > 0 else "no_clear_scene"))
    out = refs.copy()
    out["outcome"], out["n_clear_scenes"] = outcome, np.asarray(n_clear, dtype=int)
    return out
