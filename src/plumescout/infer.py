"""Inference and post-processing on Gulf S2 chips (TRD 5.6).

Flare mask: a predicted component is masked if a VIIRS 375 m fire detection lies within
max_m of any of its pixel centres and within |dt| <= max_dt of the S2 overpass. Distances
are computed in the chip's UTM CRS (metres); fire points must be projected to it first.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from affine import Affine


def pixel_centres(mask: np.ndarray, transform: Affine) -> np.ndarray:
    """(N, 2) x, y of the centres of True pixels, in the CRS of `transform`."""
    rows, cols = np.nonzero(mask)
    xs = transform.c + transform.a * (cols + 0.5) + transform.b * (rows + 0.5)
    ys = transform.f + transform.d * (cols + 0.5) + transform.e * (rows + 0.5)
    return np.column_stack([xs, ys])


def flare_flags(
    components: Sequence[np.ndarray],
    transform: Affine,
    scene_time: pd.Timestamp,
    fires: pd.DataFrame,
    max_m: float,
    max_dt: pd.Timedelta,
) -> list[bool]:
    """One flag per component: True if a fire (columns x, y in the chip CRS, datetime_utc)
    lies within max_m of the component and within max_dt of scene_time."""
    if len(fires) == 0:
        return [False] * len(components)
    f = fires[(pd.to_datetime(fires.datetime_utc, utc=True) - scene_time).abs() <= max_dt]
    if f.empty:
        return [False] * len(components)
    fxy = f[["x", "y"]].to_numpy(dtype=float)
    flags = []
    for comp in components:
        pxy = pixel_centres(comp, transform)
        d2 = ((pxy[:, None, :] - fxy[None, :, :]) ** 2).sum(-1)
        flags.append(bool(len(pxy)) and bool(d2.min() <= max_m**2))
    return flags


def fires_to_chip_crs(fires: pd.DataFrame, epsg: int) -> pd.DataFrame:
    """Add x, y (metres, EPSG `epsg`) to fire detections given lon, lat (EPSG:4326)."""
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tr.transform(fires.lon.to_numpy(), fires.lat.to_numpy())
    return fires.assign(x=x, y=y)


def chip_to_model_input(
    dn: np.ndarray,
    ref_dn: np.ndarray | None,
    chip_bands: list[str],
    model_bands: list[str],
    mapping: dict,
    dn_scale: float = 10000.0,
) -> np.ndarray:
    """Our L1C chip (13, H, W) uint16 DN -> model input (len(model_bands) + 1, H, W) float32.

    Bands: clip(slope * reflectance + offset, 0, 255) / 255 (CH4Net-like uint8 scale).
    Last channel: multi-pass MBMP dR (target vs reference chip), stretched per chip to
    [0, 1] between `mbmp_stretch_pct` percentiles; 0.5 everywhere if no reference.
    No-data pixels (DN 0) become 0 in band channels and 0.5 in MBMP.
    """
    from plumescout import mbmp

    refl = dn.astype(np.float64) / dn_scale
    valid = (dn > 0).all(axis=0)
    chans = []
    for b in model_bands:
        v = mapping["slope"][b] * refl[chip_bands.index(b)] + mapping["offset"][b]
        chans.append(np.where(valid, np.clip(v, 0, 255) / 255, 0.0))
    m = np.full(dn.shape[1:], 0.5)
    if ref_dn is not None:
        r = ref_dn.astype(np.float64) / dn_scale
        i11, i12 = chip_bands.index("B11"), chip_bands.index("B12")
        ok = valid & (ref_dn > 0).all(axis=0)
        t11, t12 = np.where(ok, refl[i11], np.nan), np.where(ok, refl[i12], np.nan)
        f11, f12 = np.where(ok, r[i11], np.nan), np.where(ok, r[i12], np.nan)
        if ok.sum() > 10:
            dr = mbmp.multi_pass(t11, t12, f11, f12)
            lo, hi = np.nanpercentile(dr, mapping["mbmp_stretch_pct"])
            if hi > lo:
                m = np.where(np.isfinite(dr), np.clip((dr - lo) / (hi - lo), 0, 1), 0.5)
    chans.append(m)
    return np.stack(chans).astype(np.float32)


def components_to_rows(
    comps: Sequence[np.ndarray],
    prob: np.ndarray | None,
    transform: Affine,
    epsg: int,
) -> list[dict]:
    """Per component: area_px, max_prob, centroid lon/lat (EPSG:4326), polygon WKT (EPSG:4326)."""
    from pyproj import Transformer
    from rasterio import features
    from shapely import transform as shp_transform
    from shapely.geometry import shape
    from shapely.ops import unary_union

    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    rows = []
    for c in comps:
        polys = [
            shape(g)
            for g, v in features.shapes(c.astype(np.uint8), mask=c, transform=transform)
            if v
        ]
        geom = shp_transform(
            unary_union(polys), lambda x, y, t=tr: t.transform(x, y), interleaved=False
        )
        cen = geom.centroid
        rows.append(
            {
                "area_px": int(c.sum()),
                "max_prob": float(prob[c].max()) if prob is not None else float("nan"),
                "lon": cen.x,
                "lat": cen.y,
                "geometry_wkt": geom.wkt,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    """CLI: run the frozen model and the MBMP baseline on every Gulf scene in a request window.

    Writes data/detections.parquet, data/detections/{site_id}.geojson (model, flare-masked
    detections dropped) and data/predictions/{site_id}/{YYYYMMDD}.npz (uint8 probability).
    """
    import argparse
    import json
    import logging

    import rasterio
    import torch
    from shapely import wkt
    from shapely.geometry import mapping

    from plumescout import mbmp, metrics
    from plumescout.config import load_yaml, resolve_path
    from plumescout.dataset import pad_to
    from plumescout.train import PlumeModule

    ap = argparse.ArgumentParser(description="Gulf inference (frozen operating point).")
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("plumescout.infer")

    cfg, scfg = load_yaml(args.config), load_yaml("sites.yaml")
    g = cfg["gulf_inference"]
    bands, chip_bands = cfg["data"]["input_bands"], scfg["sentinel2"]["bands"]
    fm = scfg["flare_mask"]
    gj = json.loads(resolve_path(scfg["paths"]["sites_geojson"]).read_text())
    site_ids = [f["properties"]["site_id"] for f in gj["features"]]
    scenes = pd.read_parquet(resolve_path(scfg["paths"]["scenes_table"]))
    scenes = scenes[scenes.site_id.isin(site_ids)].copy()
    scenes["date"] = pd.to_datetime(scenes.date).dt.date
    paths = scenes.set_index(["site_id", "date"]).chip_path
    todo = scenes[scenes.in_request_window.astype(bool)]
    fires = pd.read_parquet(resolve_path("data/firms.parquet"))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlumeModule.load_from_checkpoint(resolve_path(g["checkpoint"]), cfg=cfg).eval().to(dev)
    k = scfg["mbmp"]["threshold_k"]
    pred_dir = resolve_path("data/predictions")
    rows, scene_rows = [], []
    for s in todo.itertuples():
        with rasterio.open(s.chip_path) as src:
            dn, transform, epsg = src.read(), src.transform, src.crs.to_epsg()
        ref = None
        if pd.notna(s.ref_scene_date):
            with rasterio.open(paths[(s.site_id, pd.Timestamp(s.ref_scene_date).date())]) as src:
                ref = src.read()
        x = chip_to_model_input(dn, ref, chip_bands, bands, cfg["l1c_to_ch4net"])
        h, w = x.shape[1:]
        xp, valid = pad_to(x, -(-h // 32) * 32, -(-w // 32) * 32)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(xp)[None].to(dev)))[0, 0].cpu().numpy()
        p = p[:h, :w]
        out = pred_dir / s.site_id / f"{s.date:%Y%m%d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, prob_u8=np.round(p * 255).astype(np.uint8))
        t = pd.Timestamp(s.datetime_utc)
        f = fires_to_chip_crs(fires, epsg) if len(fires) else fires.assign(x=[], y=[])
        method_comps = {
            "unet": (metrics.components(p >= g["prob_threshold"], g["min_component_px"]), p)
        }
        ok = (dn > 0).all(0) & (ref > 0).all(0) if ref is not None else np.zeros(dn.shape[1:], bool)
        scene_rows.append(
            {
                "scene_id": s.scene_id,
                "site_id": s.site_id,
                "has_mbmp": bool(ok.sum() > 10),
                "max_prob": float(p.max()),
            }
        )
        if ok.sum() > 10:
            r11, r12 = (ref[chip_bands.index(b)] / 1e4 for b in ("B11", "B12"))
            t11, t12 = (dn[chip_bands.index(b)] / 1e4 for b in ("B11", "B12"))
            dr = mbmp.multi_pass(*(np.where(ok, a, np.nan) for a in (t11, t12, r11, r12)))
            method_comps["mbmp"] = (
                metrics.components(mbmp.threshold_mask(dr, k), g["min_component_px"]),
                None,
            )
        for method, (comps, prob) in method_comps.items():
            flags = flare_flags(
                comps,
                transform,
                t,
                f,
                fm["max_distance_m"],
                pd.Timedelta(hours=fm["max_time_delta_hours"]),
            )
            for i, (r, fl) in enumerate(
                zip(components_to_rows(comps, prob, transform, epsg), flags, strict=True)
            ):
                rows.append(
                    {
                        "det_id": f"{method}_{s.scene_id}_{i}",
                        "method": method,
                        "scene_id": s.scene_id,
                        "site_id": s.site_id,
                        "datetime_utc": t,
                        "flare_masked": fl,
                        **r,
                    }
                )
    det = pd.DataFrame(rows)
    det["model"] = g["checkpoint"]
    det["licence"] = (
        "Derived from Copernicus Sentinel-2 L1C; model trained on CH4Net (CC BY-NC-ND 4.0)"
    )
    det.to_parquet(resolve_path("data/detections.parquet"), index=False)
    pd.DataFrame(scene_rows).to_parquet(resolve_path("data/predicted_scenes.parquet"), index=False)
    ddir = resolve_path(scfg["paths"]["detections"])
    ddir.mkdir(parents=True, exist_ok=True)
    for sid in site_ids:
        dd = (
            det[(det.site_id == sid) & (det.method == "unet") & ~det.flare_masked]
            if len(det)
            else det
        )
        feats = [
            {
                "type": "Feature",
                "properties": {
                    "det_id": r.det_id,
                    "date_utc": str(r.datetime_utc.date()),
                    "area_px": r.area_px,
                    "max_prob": round(r.max_prob, 3),
                },
                "geometry": mapping(wkt.loads(r.geometry_wkt)),
            }
            for r in dd.itertuples()
        ]
        (ddir / f"{sid}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": feats})
        )
    log.info(
        "%d scenes, %d detections (%s)",
        len(todo),
        len(det),
        det.groupby("method").size().to_dict() if len(det) else {},
    )


if __name__ == "__main__":
    main()
