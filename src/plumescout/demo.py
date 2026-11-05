"""Precomputed demo assets for the Gradio app (TRD 8). No GEE calls or credentials at run time.

For each Gulf scene in a request window: four PNG panels (true colour, MBMP dR, U-Net
probability with detections, true colour with reference outlines) and a neutral verdict.
Detections are coordinates and dates only (hard rules 7 and 8): no rates, no rankings.
Output: app/assets/{site_id}/{YYYYMMDD}_{panel}.png and app/assets/index.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PANELS = ("rgb", "dr", "pred", "ref")
ATTRIBUTION = (
    "Contains modified Copernicus Sentinel-2 L1C data (ESA). Reference plumes: NASA EMIT L2B CH4 "
    "V002 (LP DAAC) and Carbon Mapper (non-commercial terms). Flares: NASA FIRMS VIIRS. Model "
    "trained on CH4Net (Vaughan et al. 2024, CC BY-NC-ND 4.0). Non-commercial portfolio project."
)


def verdict(
    n_det: int,
    statuses: list[str],
    ref_label: str | None,
    emit_tn: bool,
    clear: bool,
    in_train_overlap: bool,
) -> str:
    """Two-line, neutral verdict for one scene.

    Args:
        n_det: detections kept after the flare mask.
        statuses: match_status of those detections (tp / fp / unverified).
        ref_label: source of a reference plume within 1 km and 1 day of the site, e.g.
            "EMIT" or "Carbon Mapper (Tanager)", or None.
        emit_tn: EMIT observed the site within 1 day and reported no plume.
        clear: scene obscured fraction below the clear threshold.
        in_train_overlap: site within 2 km of a training location.
    """
    first = (
        f"Plume-like signal detected ({n_det} region{'s' if n_det != 1 else ''})."
        if n_det
        else "No plume-like signal detected."
    )
    if ref_label:
        second = f"{ref_label} reported a plume within 1 km and 1 day."
        if n_det and "tp" not in statuses:
            second += " The detected region is more than 1 km from it."
        if not n_det:
            second += " It was not detected here."
    elif emit_tn:
        second = "EMIT observed this site within 1 day and reported no plume."
    else:
        second = "No reference observation within 1 day, so this scene is not verified."
    notes = []
    if not clear:
        notes.append("Scene is partly covered by cloud or shadow; not used in the metrics.")
    if in_train_overlap:
        notes.append(
            "Site is within 2 km of a training location and is excluded from the evaluation."
        )
    return " ".join([first, second, *notes])


def _site_label(region: str, site_id: str) -> str:
    return f"{region}, {site_id}"


def build_assets(out_dir: Path) -> dict:
    """Render all panels and write index.json. Returns the index."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from pyproj import Transformer
    from shapely import wkt

    from plumescout import mbmp
    from plumescout.config import load_yaml, resolve_path
    from plumescout.sites import haversine_m

    cfg = load_yaml("sites.yaml")
    bands = cfg["sentinel2"]["bands"]
    clear_thr = cfg["sentinel2"]["max_cloud_frac_clear"]
    max_dt = pd.Timedelta(days=cfg["matching"]["max_time_delta_days"])
    max_m = cfg["matching"]["max_distance_m"]
    gj = json.loads(resolve_path(cfg["paths"]["sites_geojson"]).read_text())
    sites = pd.DataFrame([f["properties"] for f in gj["features"]]).set_index("site_id")
    sc = pd.read_parquet(resolve_path(cfg["paths"]["scenes_table"]))
    sc = sc[sc.site_id.isin(sites.index)].copy()
    sc["date"] = pd.to_datetime(sc.date).dt.date
    sc["datetime_utc"] = pd.to_datetime(sc.datetime_utc, utc=True)
    paths = sc.set_index(["site_id", "date"]).chip_path
    todo = sc[sc.in_request_window.astype(bool)].sort_values(["site_id", "date"])
    det = pd.read_parquet(resolve_path("models/gulf_eval/detections_unet.parquet"))
    refs = pd.read_parquet(resolve_path(cfg["paths"]["references_table"]))
    obs = pd.read_parquet(resolve_path("data/emit_observations.parquet"))
    tn = obs[~obs.emit_plume.astype(bool)]
    metrics = json.loads(resolve_path("models/gulf_eval/gulf_metrics.json").read_text())

    def save(img_fn, path: Path) -> None:
        fig = plt.figure(figsize=(3, 3), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        img_fn(ax)
        ax.set_xlim(-0.5, 199.5)
        ax.set_ylim(199.5, -0.5)
        ax.axis("off")
        fig.savefig(path, dpi=100)
        plt.close(fig)

    index: dict = {"attribution": ATTRIBUTION, "metrics": metrics, "sites": []}
    for sid, g in todo.groupby("site_id"):
        srow = sites.loc[sid]
        entries = []
        (out_dir / sid).mkdir(parents=True, exist_ok=True)
        for s in g.itertuples():
            with rasterio.open(s.chip_path) as src:
                dn, tr, epsg = src.read().astype(float) / 1e4, src.transform, src.crs.to_epsg()
            to_px_tr = Transformer.from_crs(4326, epsg, always_xy=True)

            def to_px(xs, ys, _t=to_px_tr, _tr=tr):
                x, y = _t.transform(np.asarray(xs), np.asarray(ys))
                inv = ~_tr
                return inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f

            dr = None
            if pd.notna(s.ref_scene_date):
                with rasterio.open(paths[(sid, pd.Timestamp(s.ref_scene_date).date())]) as src:
                    rf = src.read().astype(float) / 1e4
                ok = (dn > 0).all(0) & (rf > 0).all(0)
                if ok.sum() > 10:
                    gb = [
                        np.where(ok, a[bands.index(b)], np.nan)
                        for a in (dn, rf)
                        for b in ("B11", "B12")
                    ]
                    dr = mbmp.multi_pass(gb[0], gb[1], gb[2], gb[3])
            prob = (
                np.load(resolve_path(f"data/predictions/{sid}/{s.date:%Y%m%d}.npz"))["prob_u8"]
                / 255
            )
            rgb = np.stack([dn[bands.index(b)] for b in ("B4", "B3", "B2")], -1)
            lo, hi = np.nanpercentile(rgb, [2, 98])
            rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
            d = det[det.scene_id == s.scene_id]
            kept = d[~d.flare_masked]
            near_refs = refs[
                (refs.site_id == sid) & ((refs.datetime_utc - s.datetime_utc).abs() <= max_dt)
            ]
            near_refs = near_refs[
                haversine_m(srow.lon, srow.lat, near_refs.lon, near_refs.lat) <= max_m
            ]
            stem = f"{s.date:%Y%m%d}"

            def outline(ax, geom_wkt, color, ls="-", _px=to_px):
                geom = wkt.loads(geom_wkt)
                for gg in getattr(geom, "geoms", [geom]):
                    xs, ys = gg.exterior.xy
                    c, r = _px(xs, ys)
                    ax.plot(c, r, color=color, lw=1.5, ls=ls)

            save(lambda ax, _rgb=rgb: ax.imshow(_rgb), out_dir / sid / f"{stem}_rgb.png")

            def draw_dr(ax, _dr=dr):
                if _dr is None:
                    ax.text(100, 100, "no reference scene", ha="center", va="center")
                    return
                v = np.nanpercentile(np.abs(_dr), 98)
                ax.imshow(_dr, cmap="RdBu", vmin=-v, vmax=v)

            save(draw_dr, out_dir / sid / f"{stem}_dr.png")

            def draw_pred(ax, _p=prob, _d=d):
                ax.imshow(_p, cmap="magma", vmin=0, vmax=1)
                for r in _d.itertuples():
                    outline(
                        ax,
                        r.geometry_wkt,
                        "grey" if r.flare_masked else "cyan",
                        "--" if r.flare_masked else "-",
                    )

            save(draw_pred, out_dir / sid / f"{stem}_pred.png")

            def draw_ref(ax, _rgb=rgb, _refs=near_refs):
                ax.imshow(_rgb)
                for r in _refs.itertuples():
                    if isinstance(r.geometry_wkt, str):
                        outline(ax, r.geometry_wkt, "lime" if r.source == "emit" else "yellow")
                    c, rr = to_px([r.lon], [r.lat])
                    ax.plot(
                        c, rr, "+", color="lime" if r.source == "emit" else "yellow", ms=10, mew=2
                    )

            save(draw_ref, out_dir / sid / f"{stem}_ref.png")

            ref_label = None
            if len(near_refs):
                r0 = near_refs.iloc[0]
                inst = "EMIT retrieval" if r0.instrument.startswith("EMIT") else r0.instrument
                ref_label = "EMIT" if r0.source == "emit" else f"Carbon Mapper ({inst})"
            emit_tn = bool(
                ((tn[tn.site_id == sid].datetime_utc - s.datetime_utc).abs() <= max_dt).any()
            )
            clear = bool(s.obscured_frac < clear_thr)
            entries.append(
                {
                    "date": str(s.date),
                    "time_utc": s.datetime_utc.strftime("%H:%M"),
                    "n_det": len(kept),
                    "has_reference": ref_label is not None,
                    "verdict": verdict(
                        len(kept),
                        kept.match_status.tolist(),
                        ref_label,
                        emit_tn,
                        clear,
                        bool(srow.in_train_overlap),
                    ),
                    "files": {p: f"{sid}/{stem}_{p}.png" for p in PANELS},
                }
            )
        index["sites"].append(
            {
                "site_id": sid,
                "label": _site_label(srow.region, sid),
                "lon": float(srow.lon),
                "lat": float(srow.lat),
                "in_train_overlap": bool(srow.in_train_overlap),
                "scenes": entries,
            }
        )
        logger.info("%s: %d scenes", sid, len(entries))
    # eligible sites first, then flagged; the app opens on the first entry
    index["sites"].sort(
        key=lambda x: (x["in_train_overlap"], x["site_id"][:3] != "KWT", x["site_id"])
    )
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, default=float))
    return index


def main(argv: list[str] | None = None) -> None:
    """CLI: build app/assets."""
    import argparse

    from plumescout.config import resolve_path

    ap = argparse.ArgumentParser(description="Build Gradio demo assets.")
    ap.add_argument("--out", default="app/assets")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = build_assets(out)
    n = sum(len(s["scenes"]) for s in idx["sites"])
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    logger.info("%d sites, %d scenes, %.1f MB", len(idx["sites"]), n, size / 1e6)


if __name__ == "__main__":
    main()
