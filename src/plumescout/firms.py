"""NASA FIRMS VIIRS 375 m active fire detections for flare masking (TRD 5.6, Phase 4).

Area API: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{bbox}/{days}/{date}
with bbox = west,south,east,north and 1 to 5 days starting at `date`. Standard-processing
sources (`*_SP`) cover the archive; near-real-time (`*_NRT`) the last few months.
Times are UTC.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SOURCES_SP = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
SOURCES_NRT = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
LICENCE = "NASA FIRMS VIIRS 375 m active fire (VNP14IMG / VJ114IMG), NASA open data"


def date_windows(dates: Iterable[date], pad_days: int = 1, max_days: int = 5):
    """Merge [d - pad, d + pad] intervals and split them into API windows of <= max_days.

    Yields (start_date, n_days).
    """
    iv = sorted((d - timedelta(days=pad_days), d + timedelta(days=pad_days)) for d in dates)
    merged: list[list[date]] = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + timedelta(days=1):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        while a <= b:
            n = min(max_days, (b - a).days + 1)
            yield a, n
            a += timedelta(days=n)


def parse_csv(text: str, source: str) -> pd.DataFrame:
    """FIRMS area CSV -> rows with lon, lat, datetime_utc, source."""
    if not text.strip() or text.lstrip().startswith(("Invalid", "Error")):
        if text.strip():
            logger.warning("FIRMS %s: %s", source, text.strip()[:200])
        return pd.DataFrame(columns=["lon", "lat", "datetime_utc", "frp", "confidence", "source"])
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return pd.DataFrame(columns=["lon", "lat", "datetime_utc", "frp", "confidence", "source"])
    hhmm = df["acq_time"].astype(int).astype(str).str.zfill(4)
    t = pd.to_datetime(df["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M", utc=True)
    return pd.DataFrame(
        {
            "lon": df["longitude"].astype(float),
            "lat": df["latitude"].astype(float),
            "datetime_utc": t,
            "frp": df.get("frp", pd.Series(np.nan, index=df.index)),
            "confidence": df.get("confidence", pd.Series("", index=df.index)).astype(str),
            "source": source,
        }
    )


def fetch(
    bbox: Sequence[float], dates: Iterable[date], sources: Sequence[str] = SOURCES_SP
) -> pd.DataFrame:
    """VIIRS detections in bbox within +-1 day of each date, all sources merged."""
    import requests

    from plumescout.config import get_secret

    key = get_secret("FIRMS_MAP_KEY")
    area = ",".join(f"{v:.4f}" for v in bbox)
    frames = []
    for start, n in date_windows(dates):
        for src in sources:
            url = f"{API}/{key}/{src}/{area}/{n}/{start.isoformat()}"
            r = requests.get(url, timeout=120)
            if r.status_code != 200:  # never log the URL: it contains the MAP_KEY
                logger.warning(
                    "FIRMS %s %s +%dd: HTTP %d %s",
                    src,
                    start,
                    n,
                    r.status_code,
                    r.text.strip()[:120].replace(key, "***"),
                )
                continue
            frames.append(parse_csv(r.text, src))
    out = pd.concat(frames, ignore_index=True) if frames else parse_csv("", "")
    out["datetime_utc"] = pd.to_datetime(out.datetime_utc, utc=True)
    out["licence"] = LICENCE
    return out.drop_duplicates(["lon", "lat", "datetime_utc", "source"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    """CLI: VIIRS detections within +-1 day of every Gulf S2 scene -> data/firms.parquet."""
    import argparse
    import json

    from plumescout.config import load_yaml, resolve_path

    ap = argparse.ArgumentParser(description="Fetch FIRMS VIIRS detections for Gulf S2 dates.")
    ap.add_argument("--out", default="data/firms.parquet")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_yaml("sites.yaml")
    gj = json.loads(resolve_path(cfg["paths"]["sites_geojson"]).read_text())
    region_of = {f["properties"]["site_id"]: f["properties"]["region"] for f in gj["features"]}
    scenes = pd.read_parquet(resolve_path(cfg["paths"]["scenes_table"]))
    scenes = scenes[scenes.site_id.isin(region_of)]
    frames = []
    for name, aoi in cfg["aois"].items():
        sids = [s for s, r in region_of.items() if r == aoi["region"]]
        dates = sorted(set(scenes.loc[scenes.site_id.isin(sids), "date"]))
        if not dates:
            continue
        cache = resolve_path(f"data/raw/firms/firms_{name}.parquet")
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            df = fetch(aoi["bbox"], dates, SOURCES_SP + SOURCES_NRT)
            df["aoi"] = name
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
        logger.info("%s: %d S2 dates, %d VIIRS detections", name, len(dates), len(df))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["datetime_utc"] = pd.to_datetime(out.datetime_utc, utc=True)
    # SP and NRT can report the same detection: keep one per location and minute
    key = [out.lon.round(4), out.lat.round(4), out.datetime_utc.dt.floor("min")]
    out = out.loc[~pd.concat(key, axis=1).duplicated()].reset_index(drop=True)
    out.to_parquet(resolve_path(args.out), index=False)
    logger.info("wrote %s (%d detections)", args.out, len(out))


if __name__ == "__main__":
    main()
