"""Recover site and date for CH4Net images by matching them to our own L1C chips.

The CH4Net release (HF `av555/ch4net`) stores images as numbered uint8 arrays with no
site, date or georeferencing. Vaughan et al. (2024, Table 2) give the 23 site coordinates.
Observed structure of the test split (2021): IDs come in blocks of 73 (one relative orbit,
5-day revisit), blocks are grouped by site and, within a site, sorted negatives first,
then plume images. CH4Net frames are centred on the site at ~10 m (scale within 2%).

Method:
1. Site -> blocks: count block images whose best match is one of our cloudy dates
   (cloud patterns are local, so they only match at the right site).
2. Image -> date: Hungarian assignment on NCC of six z-scored bands plus the B12/B11 ratio,
   using the central 100 x 100 px of each frame. `conf` = assigned score minus the best
   alternative date for that image. Low-confidence dates must not be treated as known.

Band index maps: our 13-band chips are B1..B12 incl. B10 and B8A (config order);
CH4Net has 12 bands without B10: B1..B8, B8A, B9, B11, B12.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

CH4NET_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
MATCH_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BLOCK = 73
CROP = 100


def centre_crop(a: np.ndarray, n: int) -> np.ndarray:
    """Central n x n window of the last two axes."""
    h, w = a.shape[-2:]
    if h < n or w < n:
        raise ValueError(f"Array {a.shape} smaller than crop {n}")
    r0, c0 = (h - n) // 2, (w - n) // 2
    return a[..., r0 : r0 + n, c0 : c0 + n]


def zrows(x: np.ndarray) -> np.ndarray:
    """Z-score each row of a 2D array."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NCC matrix between z-scored rows of a (n, f) and b (m, f)."""
    return a @ b.T / a.shape[1]


def assign(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one assignment maximising total similarity (rows <= cols).

    Returns:
        cols: assigned column per row.
        conf: assigned similarity minus the best other column in that row.
    """
    r, c = linear_sum_assignment(-sim)
    cols = np.empty(sim.shape[0], dtype=int)
    cols[r] = c
    conf = np.array(
        [sim[i, cols[i]] - np.max(np.delete(sim[i], cols[i])) for i in range(len(cols))]
    )
    return cols, conf


def place_centred(label: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Put a CH4Net label on a chip grid of `shape`, centre to centre, cropping or zero-padding."""
    out = np.zeros(shape, dtype=label.dtype)
    h, w = label.shape
    hh_out, ww_out = shape
    dy, dx = hh_out // 2 - h // 2, ww_out // 2 - w // 2
    ys, xs = max(0, -dy), max(0, -dx)
    yd, xd = max(0, dy), max(0, dx)
    hh, ww = min(h - ys, hh_out - yd), min(w - xs, ww_out - xd)
    out[yd : yd + hh, xd : xd + ww] = label[ys : ys + hh, xs : xs + ww]
    return out


def _ours_features(chip: np.ndarray, band_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """chip: (13, H, W) DN. Returns flattened z-scored band stack and B12/B11 ratio."""
    a = centre_crop(chip.astype(np.float64), CROP)
    idx = [band_names.index(b) for b in MATCH_BANDS]
    i11, i12 = band_names.index("B11"), band_names.index("B12")
    ratio = a[i12] / np.maximum(a[i11], 1)
    return a[idx].reshape(len(idx), -1), ratio.ravel()


def _ch4net_features(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """arr: (H, W, 12) uint8. Same features as `_ours_features`."""
    a = centre_crop(np.moveaxis(arr.astype(np.float64), -1, 0), CROP)
    idx = [CH4NET_BANDS.index(b) for b in MATCH_BANDS]
    i11, i12 = CH4NET_BANDS.index("B11"), CH4NET_BANDS.index("B12")
    ratio = a[i12] / np.maximum(a[i11], 1)
    return a[idx].reshape(len(idx), -1), ratio.ravel()


def _zbands(stack: np.ndarray) -> np.ndarray:
    """z-score each band separately, then flatten: (bands, pixels) -> (bands * pixels,)."""
    return zrows(stack).ravel()


def index_site(
    site_id: str,
    scenes: pd.DataFrame,
    ch4net_split_dir: Path,
    band_names: list[str],
    n_blocks_total: int,
    cloudy_frac: float = 0.3,
    cloud_ncc: float = 0.85,
) -> pd.DataFrame:
    """Find the CH4Net blocks for a site and assign each image a date.

    Args:
        scenes: scenes.parquet rows for this site in the CH4Net year (one chip per date).
        ch4net_split_dir: e.g. data/raw/ch4net/test (with s2/ and label/ folders).
        n_blocks_total: number of 73-image blocks in the split.
    Returns:
        One row per CH4Net image: site_id, split, ch4net_id, block, plume_px, date,
        cloud_frac, scene_id, score, conf.
    """
    import rasterio

    g = scenes.sort_values("date").reset_index(drop=True)
    ob, orat = [], []
    for p in g.chip_path:
        with rasterio.open(p) as src:
            b, r = _ours_features(src.read(), band_names)
        ob.append(_zbands(b))
        orat.append(r)
    ob, orat = np.stack(ob), zrows(np.stack(orat))
    cloudy = (g.cloud_frac > cloudy_frac).to_numpy()

    feats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    hits = []
    for k in range(n_blocks_total):
        ids = [
            i
            for i in range(BLOCK * k, BLOCK * (k + 1))
            if (ch4net_split_dir / f"s2/{i}.npy").exists()
        ]
        for i in ids:
            b, r = _ch4net_features(np.load(ch4net_split_dir / f"s2/{i}.npy"))
            feats[i] = (_zbands(b), r)
        s = similarity(np.stack([feats[i][0] for i in ids]), ob)
        hits.append(int(((s.max(1) > cloud_ncc) & cloudy[s.argmax(1)]).sum()))
    n_blocks = max(1, round(len(g) / BLOCK))
    blocks = sorted(np.argsort(hits)[::-1][:n_blocks].tolist())
    logger.info(
        "%s: cloud-match hits per block (top) %s -> blocks %s",
        site_id,
        sorted(hits, reverse=True)[:4],
        blocks,
    )

    ids = [i for k in blocks for i in range(BLOCK * k, BLOCK * (k + 1)) if i in feats]
    sim = similarity(np.stack([feats[i][0] for i in ids]), ob) + similarity(
        zrows(np.stack([feats[i][1] for i in ids])), orat
    )
    cols, conf = assign(sim)
    lab = [int(np.load(ch4net_split_dir / f"label/{i}.npy").sum()) for i in ids]
    return pd.DataFrame(
        {
            "site_id": site_id,
            "split": ch4net_split_dir.name,
            "ch4net_id": ids,
            "block": [i // BLOCK for i in ids],
            "plume_px": lab,
            "date": g.date.to_numpy()[cols],
            "cloud_frac": g.cloud_frac.to_numpy()[cols],
            "scene_id": g.scene_id.to_numpy()[cols],
            "score": sim[np.arange(len(ids)), cols],
            "conf": conf,
        }
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: index CH4Net test images for sites whose chips are in scenes.parquet."""
    import argparse

    from plumescout.config import load_yaml, resolve_path

    p = argparse.ArgumentParser(description="Recover site/date for CH4Net test images.")
    p.add_argument("--sites", required=True, help="Comma-separated site IDs, e.g. T7,T14")
    p.add_argument("--split", default="test")
    p.add_argument("--year", type=int, default=2021)
    p.add_argument("--out", default="data/ch4net_index.parquet")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_yaml("sites.yaml")
    scenes = pd.read_parquet(resolve_path(cfg["paths"]["scenes_table"]))
    split_dir = resolve_path("data/raw/ch4net") / args.split
    n_ids = max(int(f.stem) for f in (split_dir / "s2").glob("*.npy")) + 1
    n_blocks = -(-n_ids // BLOCK)
    out = resolve_path(args.out)
    frames = []
    for sid in args.sites.split(","):
        s = scenes[(scenes.site_id == sid) & (pd.to_datetime(scenes.date).dt.year == args.year)]
        df = index_site(sid, s, split_dir, cfg["sentinel2"]["bands"], n_blocks)
        pos = df[df.plume_px > 0]
        logger.info(
            "%s: %d images, %d with plume (%.1f%% of %d slots); plume images with conf > 0.1: %d",
            sid,
            len(df),
            len(pos),
            100 * len(pos) / (BLOCK * df.block.nunique()),
            BLOCK * df.block.nunique(),
            int((pos.conf > 0.1).sum()),
        )
        frames.append(df)
    new = pd.concat(frames, ignore_index=True)
    if out.exists():
        old = pd.read_parquet(out)
        new = pd.concat([old[~old.ch4net_id.isin(new.ch4net_id) | (old.split != args.split)], new])
    new["source"] = "CH4Net (av555/ch4net, CC BY-NC-ND 4.0); dates inferred by image matching"
    new.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
