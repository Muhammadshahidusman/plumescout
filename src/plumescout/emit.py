"""NASA EMIT L2B methane references, version 002 only (hard rule 2), via earthaccess.

- `EMITL2BCH4PLM` (plume complexes): one GeoJSON metadata file per plume complex with the
  plume outline (EPSG:4326), UTC time, max concentration location and DAAC scene names.
- `EMITL2BCH4ENH` (enhancement granules): every EMIT scene processed for methane. A site
  inside a granule footprint with no plume complex on that overpass is an EMIT
  true-negative observation (TRD 5.7).

Emission rates are stored for reference only and are never presented as PlumeScout output
(hard rule 7). Times are UTC.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PLM = "EMITL2BCH4PLM"
ENH = "EMITL2BCH4ENH"
VERSION = "002"
LICENCE = "NASA EMIT L2B CH4 V002, LP DAAC, NASA open data (no restrictions)"


def _num(v) -> float:
    """'NA' or None -> nan, else float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def parse_plume_feature(feature: dict) -> dict:
    """One EMIT plume-complex GeoJSON feature -> reference row (EPSG:4326, UTC)."""
    from shapely.geometry import shape

    p = feature["properties"]
    return {
        "ref_id": f"emit_{p['Plume ID']}",
        "source": "emit",
        "datetime_utc": pd.Timestamp(p["UTC Time Observed"]).tz_convert("UTC"),
        "lon": float(p["Longitude of max concentration"]),
        "lat": float(p["Latitude of max concentration"]),
        "geometry_wkt": shape(feature["geometry"]).wkt,
        "max_ppm_m": _num(p.get("Max Plume Concentration (ppm m)")),
        "emission_rate_kgh": _num(p.get("Emissions Rate Estimate (kg/hr)")),
        "scene_names": ";".join(p.get("DAAC Scene Names", [])),
        "instrument": "EMIT",
        "licence": LICENCE,
    }


def read_plume_dir(path: Path) -> pd.DataFrame:
    """Read every plume-complex metadata JSON in `path` (duplicates by ref_id dropped)."""
    rows = []
    for f in sorted(Path(path).glob("*.json")):
        rows += [parse_plume_feature(ft) for ft in json.loads(f.read_text())["features"]]
    df = pd.DataFrame(rows)
    return df.drop_duplicates("ref_id").reset_index(drop=True) if len(df) else df


def in_bbox(lon: np.ndarray, lat: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """bbox = [min_lon, min_lat, max_lon, max_lat] in degrees."""
    lon, lat = np.asarray(lon), np.asarray(lat)
    return (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])


# ---------------------------------------------------------------- earthaccess edge


def _login() -> None:
    import earthaccess

    from plumescout.config import load_env

    load_env()
    earthaccess.login(strategy="environment")


def search(short_name: str, bbox: Sequence[float], start: date, end: date) -> list:
    """CMR granule search (version 002 only). Returns earthaccess DataGranule objects."""
    import earthaccess

    return earthaccess.search_data(
        short_name=short_name,
        version=VERSION,
        bounding_box=tuple(bbox),
        temporal=(start.isoformat(), end.isoformat()),
    )


def download_plume_metadata(bbox: Sequence[float], start: date, end: date, out_dir: Path) -> int:
    """Download only the small plume-complex metadata JSONs (not the COGs). Returns count."""
    import earthaccess

    _login()
    granules = search(PLM, bbox, start, end)
    links = [u for g in granules for u in g.data_links() if u.endswith(".json")]
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [u for u in links if not (out_dir / u.rsplit("/", 1)[-1]).exists()]
    if todo:
        earthaccess.download(todo, local_path=str(out_dir))
    logger.info("EMIT %s: %d granules, %d JSON (%d new)", PLM, len(granules), len(links), len(todo))
    return len(links)


def enhancement_overpasses(bbox: Sequence[float], start: date, end: date) -> pd.DataFrame:
    """EMIT methane-processed scenes intersecting bbox: granule id, UTC time, footprint WKT.

    Metadata only (no download).
    """
    from shapely.geometry import Polygon

    _login()
    rows = []
    for g in search(ENH, bbox, start, end):
        umm = g["umm"]
        t = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        polys = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"]
        for poly in polys:
            pts = [(p["Longitude"], p["Latitude"]) for p in poly["Boundary"]["Points"]]
            rows.append(
                {
                    "granule_id": umm["GranuleUR"],
                    "datetime_utc": pd.Timestamp(t).tz_convert("UTC"),
                    "footprint_wkt": Polygon(pts).wkt,
                }
            )
    return pd.DataFrame(rows)
