"""Download Sentinel-2 L1C chips from Google Earth Engine.

Collection: COPERNICUS/S2_HARMONIZED only (hard rule 1, never L2A).
Chips: 200 x 200 px, 10 m, local UTM zone of the site, all 13 bands resampled to 10 m.
Stored as uint16 DN GeoTIFFs (TOA reflectance = DN / 10000) at
`data/chips/{site_id}/{YYYYMMDD}.tif`, one row per chip in `scenes.parquet`.
Cloud fraction comes from s2cloudless probability within the chip.

CLI:
    uv run python -m plumescout.s2_download --sites T7,T14,T22 --start 2021-01-01 --end 2022-01-01
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LICENCE = "Contains modified Copernicus Sentinel data (Copernicus open licence)"


# ---------------------------------------------------------------- pure logic


@dataclass(frozen=True)
class ChipGrid:
    """Pixel grid for one chip. `transform` is a rasterio-style affine (a, b, c, d, e, f)."""

    epsg: int
    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat), incl. Norway/Svalbard exceptions."""
    zone = int((lon + 180) // 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    elif 72 <= lat < 84 and lon >= 0:
        if lon < 9:
            zone = 31
        elif lon < 21:
            zone = 33
        elif lon < 33:
            zone = 35
        elif lon < 42:
            zone = 37
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def chip_grid(lon: float, lat: float, size_px: int, pixel_m: float) -> ChipGrid:
    """Grid of size_px x size_px pixels of pixel_m metres centred on (lon, lat) in local UTM.

    The upper-left corner is snapped to a multiple of pixel_m.
    """
    from pyproj import Transformer

    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = size_px * pixel_m / 2
    x0 = np.floor((x - half) / pixel_m) * pixel_m
    y0 = np.ceil((y + half) / pixel_m) * pixel_m
    return ChipGrid(
        epsg=epsg,
        width=size_px,
        height=size_px,
        transform=(float(pixel_m), 0.0, float(x0), 0.0, -float(pixel_m), float(y0)),
    )


def pick_reference(
    scenes: Iterable[tuple[date, float]],
    target: date,
    lookback_days: int,
    max_cloud: float,
) -> date | None:
    """Most recent scene strictly before `target`, within `lookback_days` (inclusive),
    with cloud fraction below `max_cloud`. Returns None if there is none."""
    start = target - timedelta(days=lookback_days)
    ok = [d for d, cf in scenes if start <= d < target and cf < max_cloud]
    return max(ok) if ok else None


def cloud_fraction(cloud_prob: np.ndarray, threshold: float) -> float:
    """Fraction of finite pixels with s2cloudless probability (percent) >= threshold.

    Returns 1.0 if the chip has no finite pixels.
    """
    p = np.asarray(cloud_prob, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return 1.0
    return float((p[finite] >= threshold).sum() / finite.sum())


def shadow_fraction(nir: np.ndarray, nir_median: np.ndarray, ratio: float) -> float:
    """Fraction of finite pixels whose NIR (B8) reflectance is below `ratio` x the site's
    clear-sky median. s2cloudless flags clouds but not their shadows; this catches shadows
    from clouds just outside the chip. Returns 1.0 if no pixel is finite."""
    nir = np.asarray(nir, dtype=np.float64)
    med = np.asarray(nir_median, dtype=np.float64)
    finite = np.isfinite(nir) & np.isfinite(med) & (med > 0)
    if not finite.any():
        return 1.0
    return float((nir[finite] < ratio * med[finite]).sum() / finite.sum())


def obscured_fraction(cloud_frac: float, shadow_frac: float) -> float:
    """Upper bound on the obscured share of a chip: cloud plus shadow, capped at 1."""
    return float(min(1.0, cloud_frac + shadow_frac))


def add_shadow_and_refs(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per site: shadow_frac from B8 vs the median of near-cloud-free chips, obscured_frac,
    and ref_scene_date re-picked so references must also be shadow-free.

    Reads chips from `chip_path`. Returns a copy with the new columns.
    """
    import rasterio

    s2c, chip = cfg["sentinel2"], cfg["s2_chip"]
    i8 = s2c["bands"].index("B8") + 1
    out = []
    for _, g in df.groupby("site_id"):
        g = g.sort_values("date").copy()
        nir = {}
        for r in g.itertuples():
            with rasterio.open(r.chip_path) as src:
                a = src.read(i8).astype(np.float64) / chip["dn_scale"]
            a[a == 0] = np.nan
            nir[r.scene_id] = a
        base = g[g.cloud_frac < chip["median_base_max_cloud"]].scene_id
        med = np.nanmedian(np.stack([nir[s] for s in base]), axis=0)
        g["shadow_frac"] = [
            shadow_fraction(nir[s], med, chip["shadow_nir_ratio"]) for s in g.scene_id
        ]
        g["obscured_frac"] = [
            obscured_fraction(c, s) for c, s in zip(g.cloud_frac, g.shadow_frac, strict=True)
        ]
        pairs = list(zip(g["date"], g["obscured_frac"], strict=True))
        g["ref_scene_date"] = [
            pick_reference(pairs, d, s2c["reference_lookback_days"], s2c["max_cloud_frac_clear"])
            for d in g["date"]
        ]
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- GEE edge


def _init_ee() -> None:
    import ee

    from plumescout.config import get_secret

    ee.Initialize(project=get_secret("GEE_PROJECT"))


def list_scenes(lon: float, lat: float, start: date, end: date, cfg: dict) -> pd.DataFrame:
    """L1C products covering the point in [start, end), with s2cloudless joined.

    Returns columns: system_index, product_id, date (UTC), tile_cloud_pct.
    """
    import ee

    s2c = cfg["sentinel2"]
    pt = ee.Geometry.Point(lon, lat)
    col = (
        ee.ImageCollection(s2c["collection"])
        .filterBounds(pt)
        .filterDate(start.isoformat(), end.isoformat())
    )
    info = col.reduceColumns(
        ee.Reducer.toList(4),
        ["system:index", "PRODUCT_ID", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE"],
    ).getInfo()["list"]
    df = pd.DataFrame(info, columns=["system_index", "product_id", "time_ms", "tile_cloud_pct"])
    df["datetime_utc"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
    df["date"] = df["datetime_utc"].dt.date
    return df.drop(columns="time_ms").sort_values(["date", "tile_cloud_pct"]).reset_index(drop=True)


def fetch_chip(system_index: str, grid: ChipGrid, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one chip from GEE.

    Returns:
        bands: uint16 array (13, H, W) of DN in config band order (0 = no data).
        cloud_prob: float array (H, W), s2cloudless probability in percent (NaN = no data).
    """
    import ee

    s2c, chip = cfg["sentinel2"], cfg["s2_chip"]
    bands = s2c["bands"]
    img = ee.Image(f"{s2c['collection']}/{system_index}").select(bands)
    img = img.resample(chip["resampling"]).toUint16()
    prob = ee.Image(f"{chip['cloud_prob_collection']}/{system_index}").select("probability")
    stacked = img.addBands(prob.resample(chip["resampling"]).toFloat().rename("cloud_prob"))
    a, _, c, _, e, f = grid.transform
    arr = ee.data.computePixels(
        {
            "expression": stacked,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": grid.width, "height": grid.height},
                "affineTransform": {
                    "scaleX": a,
                    "shearX": 0,
                    "translateX": c,
                    "shearY": 0,
                    "scaleY": e,
                    "translateY": f,
                },
                "crsCode": f"EPSG:{grid.epsg}",
            },
        }
    )
    dn = np.stack([arr[b] for b in bands]).astype(np.uint16)
    cp = arr["cloud_prob"].astype(np.float64)
    cp[(dn == 0).all(axis=0)] = np.nan
    return dn, cp


def write_chip(path: Path, dn: np.ndarray, grid: ChipGrid, tags: dict, band_names: Sequence[str]):
    """Write a uint16 multi-band GeoTIFF with provenance tags."""
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=dn.shape[0],
        dtype="uint16",
        crs=f"EPSG:{grid.epsg}",
        transform=Affine(*grid.transform),
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(dn)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def download_site(
    site_id: str, lon: float, lat: float, start: date, end: date, cfg: dict, chips_dir: Path
) -> pd.DataFrame:
    """Download every L1C date for a site in [start, end) plus the lookback window before start.

    One chip per UTC date (lowest tile cloud first; the next product is tried if the chip
    has more than 1% no-data). Existing chip files are reused.
    Returns rows for scenes.parquet (TRD section 6) plus cloud and provenance columns.
    """
    s2c, chip = cfg["sentinel2"], cfg["s2_chip"]
    lookback = int(s2c["reference_lookback_days"])
    grid = chip_grid(lon, lat, s2c["chip_size_px"], s2c["pixel_size_m"])
    scenes = list_scenes(lon, lat, start - timedelta(days=lookback), end, cfg)
    logger.info("%s: %d L1C products, %d dates", site_id, len(scenes), scenes["date"].nunique())

    rows = []
    for d, group in scenes.groupby("date"):
        out = chips_dir / site_id / f"{d:%Y%m%d}.tif"
        for _, sc in group.iterrows():
            try:
                dn, cp = fetch_chip(sc.system_index, grid, cfg)
            except Exception as exc:  # noqa: BLE001 - GEE raises many types; log and move on
                logger.warning("%s %s %s: fetch failed: %s", site_id, d, sc.product_id, exc)
                continue
            nodata = float((dn == 0).all(axis=0).mean())
            if nodata > 0.01 and sc is not group.iloc[-1]:
                continue
            cf = cloud_fraction(cp, chip["cloud_prob_threshold"])
            write_chip(
                out,
                dn,
                grid,
                {
                    "site_id": site_id,
                    "product_id": sc.product_id,
                    "datetime_utc": sc.datetime_utc.isoformat(),
                    "collection": s2c["collection"],
                    "dn_scale": chip["dn_scale"],
                    "cloud_frac": f"{cf:.4f}",
                    "source": "Google Earth Engine",
                    "licence": LICENCE,
                },
                s2c["bands"],
            )
            rows.append(
                {
                    "scene_id": f"{site_id}_{d:%Y%m%d}",
                    "site_id": site_id,
                    "date": d,
                    "datetime_utc": sc.datetime_utc,
                    "s2_product": sc.product_id,
                    "cloud_frac": cf,
                    "nodata_frac": nodata,
                    "chip_path": out.as_posix(),
                    "epsg": grid.epsg,
                }
            )
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pairs = list(zip(df["date"], df["cloud_frac"], strict=True))
    df["ref_scene_date"] = [
        pick_reference(pairs, d, lookback, s2c["max_cloud_frac_clear"]) for d in df["date"]
    ]
    df["in_request_window"] = (df["date"] >= start) & (df["date"] < end)
    df["source"] = "COPERNICUS/S2_HARMONIZED via Google Earth Engine"
    df["licence"] = LICENCE
    return df


def update_scenes_table(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge rows into scenes.parquet, replacing rows with the same scene_id."""
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old["scene_id"].isin(new["scene_id"])], new], ignore_index=True)
    new = new.sort_values(["site_id", "date"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def main(argv: Sequence[str] | None = None) -> None:
    from plumescout.config import load_yaml, resolve_path

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sites", required=True, help="Comma-separated CH4Net site IDs, e.g. T7,T14")
    p.add_argument("--start", required=True, type=date.fromisoformat)
    p.add_argument("--end", required=True, type=date.fromisoformat, help="exclusive")
    p.add_argument("--config", default="sites.yaml")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_yaml(args.config)
    sites = pd.read_csv(resolve_path(cfg["ch4net_sites_csv"])).set_index("site_id")
    chips_dir = resolve_path(cfg["paths"]["chips"])
    table = resolve_path(cfg["paths"]["scenes_table"])

    _init_ee()
    for sid in args.sites.split(","):
        row = sites.loc[sid]
        df = download_site(sid, row.lon, row.lat, args.start, args.end, cfg, chips_dir)
        if not df.empty:
            df = add_shadow_and_refs(df, cfg)
        if not df.empty:
            update_scenes_table(df, table)
        clear = int((df["obscured_frac"] < cfg["sentinel2"]["max_cloud_frac_clear"]).sum())
        logger.info("%s: %d chips written, %d clear", sid, len(df), clear)


if __name__ == "__main__":
    main()
