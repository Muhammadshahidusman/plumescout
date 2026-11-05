"""Site selection helpers (TRD 5.1).

Coordinates are EPSG:4326 (lon, lat in degrees); distances are metres (haversine).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8


def cluster_points(lon: np.ndarray, lat: np.ndarray, eps_m: float) -> np.ndarray:
    """DBSCAN (min_samples=1) on great-circle distance; points chained within eps_m share a label.

    Returns integer labels, one per point.
    """
    from sklearn.cluster import DBSCAN

    coords = np.radians(np.column_stack([lat, lon]))
    db = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=1, metric="haversine")
    return db.fit_predict(coords)


PREFIX = {"Kuwait": "KWT", "Southern Iraq": "IRQ", "Oman": "OMN"}


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres between points given in degrees (broadcasts)."""
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def nearest_m(lon, lat, ref_lon, ref_lat) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) and index of the nearest reference point for each query point."""
    d = haversine_m(
        np.asarray(lon)[:, None],
        np.asarray(lat)[:, None],
        np.asarray(ref_lon)[None, :],
        np.asarray(ref_lat)[None, :],
    )
    return d.min(axis=1), d.argmin(axis=1)


def build_sites(
    refs: pd.DataFrame, aois: dict, eps_m: float, train: pd.DataFrame, overlap_m: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster reference plumes into sites per AOI (TRD 5.1).

    Args:
        refs: reference plumes with ref_id, source (emit / cm), instrument, datetime_utc,
            lon, lat (EPSG:4326).
        aois: {name: {region, bbox [min_lon, min_lat, max_lon, max_lat], prefix?}}.
        eps_m: DBSCAN chaining distance (1 km).
        train: training locations (lon, lat, train_source), may be empty.
        overlap_m: in_train_overlap distance (2 km, hard rule 5).
    Returns:
        sites: site_id, region, aoi, lon, lat (mean of member plumes), n_refs, n_emit, n_cm,
            n_cm_tanager, first_utc, last_utc, nearest_train_m, nearest_train_source,
            in_train_overlap, source.
        refs: input rows inside an AOI with site_id added.
    """
    sites, kept = [], []
    for name, aoi in aois.items():
        lo0, la0, lo1, la1 = aoi["bbox"]
        r = refs[refs.lon.between(lo0, lo1) & refs.lat.between(la0, la1)].copy()
        if r.empty:
            continue
        r["cluster"] = cluster_points(r.lon.to_numpy(), r.lat.to_numpy(), eps_m)
        g = r.groupby("cluster").agg(
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            n_refs=("ref_id", "size"),
            n_emit=("source", lambda s: int((s == "emit").sum())),
            n_cm=("source", lambda s: int((s == "cm").sum())),
            first_utc=("datetime_utc", "min"),
            last_utc=("datetime_utc", "max"),
        )
        g["n_cm_tanager"] = r.groupby("cluster").apply(
            lambda x: int(((x.source == "cm") & (x.instrument == "Tanager")).sum()),
            include_groups=False,
        )
        g = g.sort_values(["n_refs", "lat"], ascending=[False, False])
        prefix = aoi.get("prefix") or PREFIX.get(aoi["region"], name[:3].upper())
        g["site_id"] = [f"{prefix}{i + 1:02d}" for i in range(len(g))]
        g["region"], g["aoi"] = aoi["region"], name
        r["site_id"] = r.cluster.map(g.site_id)
        sites.append(g.reset_index(drop=True))
        kept.append(r.drop(columns="cluster"))
    if not sites:
        return pd.DataFrame(), refs.iloc[0:0]
    s = pd.concat(sites, ignore_index=True)
    s["source"] = np.where(s.n_emit > 0, np.where(s.n_cm > 0, "emit+cm", "emit"), "cm")
    if len(train):
        d, i = nearest_m(s.lon, s.lat, train.lon, train.lat)
        s["nearest_train_m"] = d.round(0)
        s["nearest_train_source"] = train.train_source.to_numpy()[i]
    else:
        s["nearest_train_m"], s["nearest_train_source"] = np.inf, None
    s["in_train_overlap"] = s.nearest_train_m <= overlap_m
    return s, pd.concat(kept, ignore_index=True)


def emit_observations(
    sites: pd.DataFrame,
    overpasses: pd.DataFrame,
    plumes: pd.DataFrame,
    radius_m: float,
    max_dt: pd.Timedelta,
) -> pd.DataFrame:
    """EMIT overpasses that observed each site, and whether EMIT reported a plume there.

    A site is observed if it lies inside the granule footprint. `emit_plume` is True if an
    EMIT plume complex lies within radius_m and |dt| <= max_dt of the overpass; otherwise
    the overpass is an EMIT true-negative observation (TRD 5.7).
    Returns: site_id, granule_id, datetime_utc, emit_plume.
    """
    from shapely import wkt
    from shapely.geometry import Point

    rows = []
    pts = {r.site_id: Point(r.lon, r.lat) for r in sites.itertuples()}
    site_ll = sites.set_index("site_id")[["lon", "lat"]]
    for o in overpasses.itertuples():
        poly = wkt.loads(o.footprint_wkt)
        for sid, p in pts.items():
            if not poly.contains(p):
                continue
            near = plumes[(plumes.datetime_utc - o.datetime_utc).abs() <= max_dt]
            hit = bool(len(near)) and bool(
                (
                    haversine_m(
                        site_ll.loc[sid, "lon"], site_ll.loc[sid, "lat"], near.lon, near.lat
                    )
                    <= radius_m
                ).any()
            )
            rows.append(
                {
                    "site_id": sid,
                    "granule_id": o.granule_id,
                    "datetime_utc": o.datetime_utc,
                    "emit_plume": hit,
                }
            )
    cols = ["site_id", "granule_id", "datetime_utc", "emit_plume"]
    out = pd.DataFrame(rows, columns=cols)
    return out.drop_duplicates(["site_id", "granule_id"]).reset_index(drop=True)


def target_dates(times: pd.Series, pad_days: int = 1) -> list:
    """Sorted unique UTC dates within +-pad_days of each timestamp."""
    days = set()
    for t in pd.to_datetime(times, utc=True):
        for k in range(-pad_days, pad_days + 1):
            days.add((t + pd.Timedelta(days=k)).date())
    return sorted(days)


def select_sites(candidates: pd.DataFrame, min_refs: int, flagged_per_region: int) -> pd.DataFrame:
    """Eligible sites (no train overlap, >= min_refs references) plus the top
    `flagged_per_region` overlapping sites per region by reference count."""
    elig = candidates[~candidates.in_train_overlap & (candidates.n_refs >= min_refs)]
    flag = (
        candidates[candidates.in_train_overlap]
        .sort_values("n_refs", ascending=False)
        .groupby("region")
        .head(flagged_per_region)
    )
    return pd.concat([elig, flag]).reset_index(drop=True)


def train_locations(cfg: dict) -> pd.DataFrame:
    """CH4Net and MARS-S2L train/val locations (lon, lat, train_source), EPSG:4326."""
    from plumescout.config import resolve_path

    ch = pd.read_csv(resolve_path(cfg["ch4net_sites_csv"])).assign(train_source="CH4Net")
    ms = pd.read_csv(resolve_path(cfg["site_selection"]["marss2l_locations_csv"])).assign(
        train_source="MARS-S2L"
    )
    cols = ["lon", "lat", "train_source"]
    return pd.concat([ch[cols], ms[cols]], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """CLI: build references.parquet, sites.geojson and emit_observations.parquet."""
    import argparse
    import json
    import logging
    from datetime import date

    from plumescout import carbonmapper, emit
    from plumescout.config import load_yaml, resolve_path

    ap = argparse.ArgumentParser(description="Build the Gulf site list and reference tables.")
    ap.add_argument("--refresh-emit", action="store_true", help="re-query EMIT V002 via CMR")
    ap.add_argument("--summary", action="store_true", help="print the Phase 3 exit table only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("plumescout.sites")

    cfg = load_yaml("sites.yaml")
    if args.summary:
        gj = json.loads(resolve_path(cfg["paths"]["sites_geojson"]).read_text())
        sites = pd.DataFrame([f["properties"] for f in gj["features"]])
        refs = pd.read_parquet(resolve_path(cfg["paths"]["references_table"]))
        obs = pd.read_parquet(resolve_path("data/emit_observations.parquet"))
        scenes = pd.read_parquet(resolve_path(cfg["paths"]["scenes_table"]))
        scenes = scenes[scenes.site_id.isin(sites.site_id)]
        table, pairs = site_table(
            sites,
            refs,
            obs,
            scenes,
            pd.Timedelta(days=cfg["matching"]["max_time_delta_days"]),
            cfg["sentinel2"]["max_cloud_frac_clear"],
        )
        pairs.to_parquet(resolve_path("data/reference_scene_pairs.parquet"), index=False)
        table.to_csv(resolve_path("data/sites/phase3_exit_table.csv"), index=False)
        print(table.to_string(index=False))
        return
    sel_cfg, raw = cfg["site_selection"], resolve_path(cfg["paths"]["raw"])
    plm_dir = raw / "emit" / "EMITL2BCH4PLM_002"
    start, end = (date.fromisoformat(d) for d in sel_cfg["emit_period"])
    overpasses = []
    for name, aoi in cfg["aois"].items():
        f = raw / "emit" / f"enh_overpasses_{name}.parquet"
        if args.refresh_emit or not f.exists():
            emit.download_plume_metadata(aoi["bbox"], start, end, plm_dir)
            emit.enhancement_overpasses(aoi["bbox"], start, end).to_parquet(f)
        overpasses.append(pd.read_parquet(f))
    overpasses = pd.concat(overpasses, ignore_index=True)

    refs = pd.concat(
        [emit.read_plume_dir(plm_dir), carbonmapper.read_exports(raw / "carbonmapper")],
        ignore_index=True,
    )
    cl = cfg["clustering"]
    cand, refs_in = build_sites(
        refs, cfg["aois"], cl["dbscan_eps_m"], train_locations(cfg), cl["train_overlap_radius_m"]
    )
    sites = select_sites(cand, sel_cfg["min_refs"], sel_cfg["flagged_per_region"])
    obs = emit_observations(
        sites,
        overpasses,
        refs_in[refs_in.source == "emit"],
        cfg["matching"]["max_distance_m"],
        pd.Timedelta(hours=sel_cfg["emit_overpass_max_dt_hours"]),
    )
    obs["licence"] = emit.LICENCE

    refs_out = refs_in[refs_in.site_id.isin(sites.site_id)].copy()
    refs_out["date"] = refs_out.datetime_utc.dt.date
    refs_out.to_parquet(resolve_path(cfg["paths"]["references_table"]), index=False)
    cand.to_csv(resolve_path("data/sites/candidates.csv"), index=False)
    obs.to_parquet(resolve_path("data/emit_observations.parquet"), index=False)

    feats = []
    for r in sites.itertuples():
        props = {
            "site_id": r.site_id,
            "region": r.region,
            "lon": round(r.lon, 6),
            "lat": round(r.lat, 6),
            "source": r.source,
            "in_train_overlap": bool(r.in_train_overlap),
            "nearest_train_m": float(r.nearest_train_m),
            "nearest_train_source": r.nearest_train_source,
            "n_emit_plumes": int(r.n_emit),
            "n_cm_plumes": int(r.n_cm),
        }
        feats.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [props["lon"], props["lat"]]},
            }
        )
    gj = {
        "type": "FeatureCollection",
        "name": "plumescout_gulf_sites",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "metadata": {
            "built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "sources": [
                emit.LICENCE,
                carbonmapper.LICENCE,
                "Training locations: CH4Net (Vaughan et al. 2024), MARS-S2L CC BY-NC-SA 4.0",
            ],
            "note": "Sites are reference plume cluster coordinates; no rates or rankings.",
        },
        "features": feats,
    }
    out = resolve_path(cfg["paths"]["sites_geojson"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gj, indent=1))
    log.info(
        "%d candidates, %d selected (%d flagged in_train_overlap), %d references, "
        "%d EMIT observations",
        len(cand),
        len(sites),
        int(sites.in_train_overlap.sum()),
        len(refs_out),
        len(obs),
    )


def pair_with_scenes(
    events: pd.DataFrame, scenes: pd.DataFrame, max_dt: pd.Timedelta
) -> pd.DataFrame:
    """For each event (site_id, datetime_utc), count S2 scenes of the same site within
    |dt| <= max_dt (UTC datetimes), and how many are clear (`clear` column of scenes).

    Returns `events` with n_scenes, n_clear and clear_scene_ids (';'-joined) added.
    """
    out = events.copy()
    n, nc, ids = [], [], []
    by_site = {k: g for k, g in scenes.groupby("site_id")}
    for e in events.itertuples():
        g = by_site.get(e.site_id)
        if g is None:
            n.append(0), nc.append(0), ids.append("")
            continue
        m = (g.datetime_utc - e.datetime_utc).abs() <= max_dt
        c = m & g.clear
        n.append(int(m.sum())), nc.append(int(c.sum())), ids.append(";".join(g.scene_id[c]))
    out["n_scenes"], out["n_clear"], out["clear_scene_ids"] = n, nc, ids
    return out


def site_table(
    sites: pd.DataFrame,
    refs: pd.DataFrame,
    obs: pd.DataFrame,
    scenes: pd.DataFrame,
    max_dt: pd.Timedelta,
    max_obscured: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 3 exit table: per site, reference counts and matched clear S2 scenes.

    Args:
        sites: site_id, region, in_train_overlap.
        refs: references (source emit / cm, instrument, site_id, datetime_utc).
        obs: EMIT observations (site_id, datetime_utc, emit_plume).
        scenes: S2 scenes (site_id, scene_id, datetime_utc, obscured_frac).
        max_dt: S2-reference matching window (+-1 day).
        max_obscured: clear-scene threshold (cloud + shadow fraction).
    Returns:
        table: one row per site (+ TOTAL row).
        pairs: every EMIT observation and Carbon Mapper plume with its matched scenes.
    """
    sc = scenes.assign(clear=scenes.obscured_frac < max_obscured)
    sc["datetime_utc"] = pd.to_datetime(sc.datetime_utc, utc=True)
    emit_ev = obs[["site_id", "datetime_utc", "emit_plume"]].assign(kind="emit")
    cm_ev = refs[refs.source == "cm"][["site_id", "datetime_utc", "instrument"]].assign(kind="cm")
    pairs = pair_with_scenes(pd.concat([emit_ev, cm_ev], ignore_index=True), sc, max_dt=max_dt)
    rows = []
    for s in sites.itertuples():
        p = pairs[pairs.site_id == s.site_id]
        e, c = p[p.kind == "emit"], p[p.kind == "cm"]
        rows.append(
            {
                "site_id": s.site_id,
                "region": s.region,
                "in_train_overlap": bool(s.in_train_overlap),
                "emit_plumes": int((refs.site_id.eq(s.site_id) & refs.source.eq("emit")).sum()),
                "emit_plume_overpasses": int(e.emit_plume.sum()),
                "emit_tn_overpasses": int((~e.emit_plume.astype(bool)).sum()),
                "cm_plumes": len(c),
                "cm_tanager": int((c.instrument == "Tanager").sum()),
                "s2_chips": int((sc.site_id == s.site_id).sum()),
                "emit_pairs_clear_plume": int(((e.n_clear > 0) & e.emit_plume.astype(bool)).sum()),
                "emit_pairs_clear_tn": int(((e.n_clear > 0) & ~e.emit_plume.astype(bool)).sum()),
                "cm_pairs_clear": int((c.n_clear > 0).sum()),
            }
        )
    t = pd.DataFrame(rows)
    total = t.drop(columns=["site_id", "region", "in_train_overlap"]).sum()
    t.loc[len(t)] = {"site_id": "TOTAL", "region": "", "in_train_overlap": False, **total}
    return t, pairs


if __name__ == "__main__":
    main()
