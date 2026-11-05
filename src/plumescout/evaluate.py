"""Gulf evaluation metrics (Phase 4, TRD 7). Bootstrap CIs resample sites (hard rule 10).

Definitions:
- plume recall = reference plumes detected / (detected + fn); references with no clear
  S2 scene within +-1 day are not counted.
- precision = tp detections / (tp + fp); `unverified` detections are not counted.
- FP per 100 clear TN scenes = fp / clear S2 scenes within +-1 day of an EMIT
  true-negative overpass * 100.
Flare-masked detections are removed before counting (TRD 5.6) and reported separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNT_COLS = ["detected", "fn", "tp_dets", "fp_dets", "clear_tn_scenes"]


def per_site_counts(
    dets: pd.DataFrame, ref_out: pd.DataFrame, clear_tn: pd.DataFrame, site_ids: list[str]
) -> pd.DataFrame:
    """Per site counts. dets: site_id, match_status, flare_masked (one method).
    ref_out: site_id, outcome. clear_tn: one row per clear TN-covered scene (site_id)."""
    rows = []
    for s in site_ids:
        d = dets[dets.site_id == s]
        kept = d[~d.flare_masked.astype(bool)]
        r = ref_out[ref_out.site_id == s]
        rows.append(
            {
                "site_id": s,
                "detected": int((r.outcome == "detected").sum()),
                "fn": int((r.outcome == "fn").sum()),
                "no_clear_scene": int((r.outcome == "no_clear_scene").sum()),
                "tp_dets": int((kept.match_status == "tp").sum()),
                "fp_dets": int((kept.match_status == "fp").sum()),
                "unverified": int((kept.match_status == "unverified").sum()),
                "flare_masked": int(d.flare_masked.astype(bool).sum()),
                "clear_tn_scenes": int((clear_tn.site_id == s).sum()),
            }
        )
    return pd.DataFrame(rows)


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def pooled(per_site: pd.DataFrame) -> dict:
    """Pooled metrics over the given sites, with counts."""
    t = per_site[COUNT_COLS].sum()
    rec = _ratio(t.detected, t.detected + t.fn)
    prec = _ratio(t.tp_dets, t.tp_dets + t.fp_dets)
    f1 = _ratio(2 * rec * prec, rec + prec) if rec == rec and prec == prec else float("nan")
    return {
        "n_sites": len(per_site),
        "n_ref_plumes": int(t.detected + t.fn),
        "n_detected": int(t.detected),
        "n_tp_dets": int(t.tp_dets),
        "n_fp_dets": int(t.fp_dets),
        "n_clear_tn_scenes": int(t.clear_tn_scenes),
        "plume_recall": rec,
        "precision": prec,
        "f1": f1,
        "fp_per_100_clear_tn": _ratio(100 * t.fp_dets, t.clear_tn_scenes),
    }


def site_bootstrap(
    per_site: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling whole sites with replacement."""
    rng = np.random.default_rng(seed)
    keys = ("plume_recall", "precision", "f1", "fp_per_100_clear_tn")
    draws = {k: [] for k in keys}
    n = len(per_site)
    for _ in range(n_boot):
        m = pooled(per_site.iloc[rng.integers(0, n, n)])
        for k in keys:
            draws[k].append(m[k])
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    out = {}
    for k in keys:
        v = np.asarray(draws[k], dtype=float)
        out[k] = (
            tuple(float(x) for x in np.nanpercentile(v, q))
            if np.isfinite(v).any()
            else (float("nan"), float("nan"))
        )
    return out


def fp_cause(fire_within_2km: bool, cloud_or_shadow: float, b12_rank: float) -> str:
    """Heuristic FP cause, fixed before Gulf evaluation (checked in this order):
    VIIRS fire within 2 km / 12 h (beyond the 750 m mask) -> flare_nearby; scene cloud or
    shadow fraction > 1% -> cloud_edge; mean B12 under the detection above the chip's 90th
    percentile -> bright_surface; else unknown."""
    if fire_within_2km:
        return "flare_nearby"
    if cloud_or_shadow > 0.01:
        return "cloud_edge"
    if b12_rank > 0.9:
        return "bright_surface"
    return "unknown"


def main(argv: list[str] | None = None) -> None:
    """CLI: match detections, compute Gulf metrics per method -> models/gulf_eval/."""
    import argparse
    import json
    import logging

    import rasterio
    from pyproj import Transformer
    from rasterio import features as rf
    from shapely import transform as shp_transform
    from shapely import wkt

    from plumescout import match
    from plumescout.config import load_yaml, resolve_path
    from plumescout.sites import haversine_m

    ap = argparse.ArgumentParser(description="Gulf evaluation.")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("plumescout.evaluate")

    cfg = load_yaml("sites.yaml")
    mcfg = cfg["matching"]
    max_m, max_dt = mcfg["max_distance_m"], pd.Timedelta(days=mcfg["max_time_delta_days"])
    clear_thr = cfg["sentinel2"]["max_cloud_frac_clear"]
    gj = json.loads(resolve_path(cfg["paths"]["sites_geojson"]).read_text())
    sites = pd.DataFrame([f["properties"] for f in gj["features"]])
    eligible = sites.loc[~sites.in_train_overlap, "site_id"].tolist()
    refs = pd.read_parquet(resolve_path(cfg["paths"]["references_table"]))
    obs = pd.read_parquet(resolve_path("data/emit_observations.parquet"))
    det = pd.read_parquet(resolve_path("data/detections.parquet"))
    fires = pd.read_parquet(resolve_path("data/firms.parquet"))
    sc = pd.read_parquet(resolve_path(cfg["paths"]["scenes_table"]))
    sc = sc[sc.site_id.isin(sites.site_id) & sc.in_request_window.astype(bool)].copy()
    sc["datetime_utc"] = pd.to_datetime(sc.datetime_utc, utc=True)
    sc["clear"] = sc.obscured_frac < clear_thr
    scene_info = sc.set_index("scene_id")

    tn = obs[~obs.emit_plume.astype(bool)]
    clear_tn = []
    for s in sc[sc.clear].itertuples():
        t = tn[tn.site_id == s.site_id]
        if ((t.datetime_utc - s.datetime_utc).abs() <= max_dt).any():
            clear_tn.append({"site_id": s.site_id, "scene_id": s.scene_id})
    clear_tn = pd.DataFrame(clear_tn, columns=["site_id", "scene_id"])

    out_dir = resolve_path("models/gulf_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in sorted(det.method.unique()):
        d = det[det.method == method].copy()
        d = match.classify_detections(d, refs, obs, max_m, max_dt)
        d["scene_clear"] = d.scene_id.map(scene_info.clear).astype(bool)
        # FP cause (heuristic, see fp_cause)
        causes = []
        for r in d.itertuples():
            if r.match_status != "fp":
                causes.append(None)
                continue
            s = scene_info.loc[r.scene_id]
            f = fires[(fires.datetime_utc - r.datetime_utc).abs() <= pd.Timedelta(hours=12)]
            near = bool(len(f)) and bool((haversine_m(r.lon, r.lat, f.lon, f.lat) <= 2000).any())
            with rasterio.open(s.chip_path) as src:
                b12 = src.read(cfg["sentinel2"]["bands"].index("B12") + 1).astype(float)
                geom = wkt.loads(r.geometry_wkt)

                tr = Transformer.from_crs(4326, src.crs.to_epsg(), always_xy=True)
                m = rf.geometry_mask(
                    [shp_transform(geom, lambda x, y, t=tr: t.transform(x, y), interleaved=False)],
                    b12.shape,
                    src.transform,
                    invert=True,
                )
            rank = float((b12[b12 > 0] < b12[m].mean()).mean()) if m.any() else 0.0
            causes.append(fp_cause(near, float(s.cloud_frac + s.shadow_frac), rank))
        d["fp_cause"] = causes
        d.to_parquet(out_dir / f"detections_{method}.parquet", index=False)
        d_clear = d[d.scene_clear]
        ro = match.reference_outcomes(refs, d_clear[~d_clear.flare_masked], sc, max_m, max_dt)
        ro.to_parquet(out_dir / f"references_{method}.parquet", index=False)
        ps = per_site_counts(d_clear, ro, clear_tn, sites.site_id.tolist())
        ps["in_train_overlap"] = ps.site_id.isin(sites.loc[sites.in_train_overlap, "site_id"])
        ps.to_csv(out_dir / f"per_site_{method}.csv", index=False)
        pe = ps[ps.site_id.isin(eligible)]
        res = {
            "eligible": pooled(pe),
            "ci95_site_bootstrap": site_bootstrap(pe, args.n_boot, seed=42),
            "excluded_overlap_sites": sorted(set(sites.site_id) - set(eligible)),
            "excluded_overlap": pooled(ps[~ps.site_id.isin(eligible)]),
            "flare_masked_dets": int(d.flare_masked.sum()),
            "unverified_dets_clear": int((d_clear.match_status == "unverified").sum()),
            "dets_on_cloudy_scenes": int((~d.scene_clear).sum()),
            "fp_by_cause_eligible": d_clear[
                d_clear.site_id.isin(eligible)
                & ~d_clear.flare_masked
                & (d_clear.match_status == "fp")
            ]
            .fp_cause.value_counts()
            .to_dict(),
            "by_reference_source_eligible": {
                src: pooled(per_site_counts(d_clear, ro[ro.source == src], clear_tn, eligible))[
                    "plume_recall"
                ]
                for src in ("emit", "cm")
            },
        }
        results[method] = res
        log.info("%s: %s", method, json.dumps(res["eligible"], default=float))
    results["rules"] = {
        "match": f"{max_m} m, +-{max_dt}",
        "clear": f"obscured_frac < {clear_thr}",
        "counted_detections": "clear scenes only, flare-masked removed",
        "ci": f"site bootstrap, {args.n_boot} resamples, eligible sites only",
    }
    (out_dir / "gulf_metrics.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
