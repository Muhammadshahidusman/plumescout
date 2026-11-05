"""Carbon Mapper plume references from portal / API CSV exports in data/raw/carbonmapper/.

Raw exports are never committed (hard rule 9, non-commercial terms). `emission_auto`
is stored as a reference rate only, never presented as PlumeScout output (hard rule 7).
Instruments: `tan` (Planet Tanager) is independent of EMIT; `emi` is Carbon Mapper's
own retrieval from EMIT data, so it can repeat an EMIT V002 plume of the same overpass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LICENCE = "Carbon Mapper data, non-commercial terms (data.carbonmapper.org); not redistributed"
INSTRUMENTS = {"tan": "Tanager", "emi": "EMIT (Carbon Mapper retrieval)", "av3": "AVIRIS-3"}


def parse_export(df: pd.DataFrame) -> pd.DataFrame:
    """Carbon Mapper plume CSV rows -> reference rows (EPSG:4326, UTC)."""
    from shapely.geometry import box

    def bounds_wkt(s) -> str | None:
        if not isinstance(s, str):
            return None
        x0, y0, x1, y1 = (float(v) for v in s.strip("[]").split(","))
        return box(x0, y0, x1, y1).wkt

    out = pd.DataFrame(
        {
            "ref_id": "cm_" + df["plume_id"].astype(str),
            "source": "cm",
            "datetime_utc": pd.to_datetime(df["datetime"], utc=True, format="ISO8601"),
            "lon": df["plume_longitude"].astype(float),
            "lat": df["plume_latitude"].astype(float),
            "geometry_wkt": df["plume_bounds"].map(bounds_wkt),
            "emission_rate_kgh": pd.to_numeric(df["emission_auto"], errors="coerce"),
            "instrument": df["instrument"].map(INSTRUMENTS).fillna(df["instrument"]),
            "licence": LICENCE,
        }
    )
    return out.drop_duplicates("ref_id").reset_index(drop=True)


def read_exports(path: Path) -> pd.DataFrame:
    """Read and merge every CSV export in `path`."""
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        return pd.DataFrame()
    out = parse_export(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    logger.info("Carbon Mapper: %d plumes from %d files", len(out), len(files))
    return out
