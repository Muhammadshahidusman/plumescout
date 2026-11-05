"""Site ids for every CH4Net image (train, val and test), needed for site-disjoint splits.

The CH4Net release (HF `av555/ch4net`) has no site or date metadata. Observed structure
(found by inspection, 2026-11-05):
- Frame shape (H x W) takes 7 values and changes only at site boundaries, and all three
  splits show the same sequence of 18 shape runs. So the splits list the same 34
  site-orbit units in the same order.
- Test (2021) ids come in 34 blocks of 73 (one per site-orbit). T1 to T11 have two
  orbits (two blocks), T12 to T23 one: 11 * 2 + 12 = 34.

Method:
1. Test blocks -> sites: NCC between each block's per-pixel median image and a clear-sky
   median template from our own L1C chips (central 100 x 100 px, six z-scored bands),
   then a capacity-constrained assignment (two blocks for T1 to T11, one for the rest).
2. Train and val: split each split's ids into shape runs, give each run the ordered list
   of sites of the matching test run, and place the site boundaries inside the run with
   dynamic programming on per-image NCC to CH4Net-domain site templates (median of that
   site's test images). Segments are contiguous and keep the test order.
Checks: per-site plume rate against Vaughan et al. (2024) Table 2, and within each site
CH4Net lists negatives before plume images (`label_order_violations`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from plumescout.ch4net_index import BLOCK, CH4NET_BANDS, CROP, MATCH_BANDS, centre_crop, zrows

logger = logging.getLogger(__name__)


def shape_runs(shapes: list) -> list[tuple[int, int]]:
    """(start, end) index ranges of consecutive equal values, end exclusive."""
    runs, start = [], 0
    for i in range(1, len(shapes) + 1):
        if i == len(shapes) or shapes[i] != shapes[start]:
            runs.append((start, i))
            start = i
    return runs


def dedupe_consecutive(seq: list) -> list:
    """Drop consecutive repeats: [a, a, b, a] -> [a, b, a]."""
    return [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]


def segment_ordered(scores: np.ndarray) -> np.ndarray:
    """Split n items into k contiguous, non-empty segments in fixed order 0..k-1,
    maximising the summed score of each item for its segment.

    Args:
        scores: (n, k) item-to-segment scores.
    Returns:
        (n,) segment index per item.
    """
    n, k = scores.shape
    if n < k:
        raise ValueError(f"{n} items cannot fill {k} non-empty segments")
    dp = np.full((n, k), -np.inf)
    stay = np.zeros((n, k), dtype=bool)
    dp[0, 0] = scores[0, 0]
    for i in range(1, n):
        for j in range(min(i + 1, k)):
            a = dp[i - 1, j]
            b = dp[i - 1, j - 1] if j > 0 else -np.inf
            stay[i, j] = a >= b
            dp[i, j] = scores[i, j] + max(a, b)
    seg = np.empty(n, dtype=int)
    j = k - 1
    for i in range(n - 1, -1, -1):
        seg[i] = j
        if i > 0 and not stay[i, j]:
            j -= 1
    return seg


def assign_blocks(sim: np.ndarray, sites: list[str], capacity: dict[str, int]) -> list[str]:
    """Assign each block (row of `sim`, blocks x sites) a site so that site s gets exactly
    capacity[s] blocks, maximising total similarity."""
    cols = [s for s in sites for _ in range(capacity[s])]
    if len(cols) != sim.shape[0]:
        raise ValueError(f"capacities sum to {len(cols)}, but there are {sim.shape[0]} blocks")
    simx = sim[:, [sites.index(c) for c in cols]]
    r, c = linear_sum_assignment(-simx)
    out = [""] * sim.shape[0]
    for i, j in zip(r, c, strict=True):
        out[i] = cols[j]
    return out


def label_order_violations(has_plume: list) -> int:
    """Negatives that come after the first plume image (0 when sorted negatives-first)."""
    seen, bad = False, 0
    for p in has_plume:
        seen |= bool(p)
        bad += seen and not p
    return bad


def _zflat(stack: np.ndarray) -> np.ndarray:
    """(bands, H, W) -> z-score each band, flatten."""
    return zrows(stack.reshape(stack.shape[0], -1)).ravel().astype(np.float32)


def ch4net_crop(path: Path) -> np.ndarray:
    """(bands, CROP, CROP) float central crop of the MATCH_BANDS of a CH4Net s2 array."""
    bi = [CH4NET_BANDS.index(b) for b in MATCH_BANDS]
    a = np.load(path, mmap_mode="r")[..., bi]
    return centre_crop(np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0), CROP)


def chip_template(chip_paths: list[str], band_names: list[str]) -> np.ndarray:
    """Clear-sky median template from our L1C chips, as a z-scored flat vector."""
    import rasterio

    bi = [band_names.index(b) for b in MATCH_BANDS]
    stack = []
    for p in chip_paths:
        with rasterio.open(p) as src:
            stack.append(centre_crop(src.read().astype(np.float64)[bi], CROP))
    return _zflat(np.median(stack, axis=0))


def build_index(
    root: Path, scenes: pd.DataFrame, band_names: list[str], max_obscured: float = 0.05
) -> pd.DataFrame:
    """Site id for every CH4Net image.

    Args:
        root: CH4Net folder with train/, val/, test/ (each with s2/ and label/).
        scenes: our scenes.parquet rows for 2021 (chips used for site templates).
    Returns:
        One row per image: folder, image_id, site_id, unit (test block order),
        has_plume, ncc (to the CH4Net-domain template of its site), method.
    """
    clear = scenes[scenes.obscured_frac < max_obscured]
    sites = sorted(clear.site_id.unique(), key=lambda s: int(s[1:]))
    tmpl = np.stack([chip_template(clear[clear.site_id == s].chip_path, band_names) for s in sites])

    test_ids = sorted(int(f.stem) for f in (root / "test/s2").glob("*.npy"))
    n_blocks = -(-(max(test_ids) + 1) // BLOCK)
    blocks = [[i for i in test_ids if i // BLOCK == k] for k in range(n_blocks)]
    crops = {i: ch4net_crop(root / f"test/s2/{i}.npy") for i in test_ids}
    bvec = np.stack([_zflat(np.median([crops[i] for i in b], axis=0)) for b in blocks])
    sim = bvec @ tmpl.T / bvec.shape[1]
    cap = {s: 2 if int(s[1:]) <= 11 else 1 for s in sites}
    block_site = assign_blocks(sim, sites, cap)
    top = sim.argmax(1)
    disagree = [k for k in range(n_blocks) if sites[top[k]] != block_site[k]]
    if disagree:
        logger.warning("blocks where best match != assignment: %s", disagree)
    srt = np.sort(sim, axis=1)
    logger.info(
        "test blocks: min NCC margin to next site %.3f", float((srt[:, -1] - srt[:, -2]).min())
    )

    # CH4Net-domain templates: median of each site's test images
    site_tmpl = {
        s: _zflat(
            np.median(
                [crops[i] for k, b in enumerate(blocks) if block_site[k] == s for i in b], axis=0
            )
        )
        for s in sites
    }
    rows = [
        {
            "folder": "test",
            "image_id": i,
            "site_id": block_site[k],
            "unit": k,
            "method": "test block template match",
        }
        for k, b in enumerate(blocks)
        for i in b
    ]
    test_shapes = [np.load(root / f"test/s2/{b[0]}.npy", mmap_mode="r").shape[:2] for b in blocks]
    test_runs = shape_runs(test_shapes)
    for folder in ("train", "val"):
        ids = sorted(int(f.stem) for f in (root / folder / "s2").glob("*.npy"))
        shapes = [np.load(root / f"{folder}/s2/{i}.npy", mmap_mode="r").shape[:2] for i in ids]
        runs = shape_runs(shapes)
        if len(runs) != len(test_runs):
            raise ValueError(f"{folder}: {len(runs)} shape runs, test has {len(test_runs)}")
        for (a, b), (ta, tb) in zip(runs, test_runs, strict=True):
            if shapes[a] != test_shapes[ta]:
                raise ValueError(f"{folder}: run shape {shapes[a]} != test {test_shapes[ta]}")
            order = dedupe_consecutive(block_site[ta:tb])
            run_ids = ids[a:b]
            if len(order) == 1:
                seg = np.zeros(len(run_ids), dtype=int)
            else:
                v = np.stack([_zflat(ch4net_crop(root / f"{folder}/s2/{i}.npy")) for i in run_ids])
                t = np.stack([site_tmpl[s] for s in order])
                seg = segment_ordered(v @ t.T / v.shape[1])
            for i, j in zip(run_ids, seg, strict=True):
                units = [k for k in range(ta, tb) if block_site[k] == order[j]]
                rows.append(
                    {
                        "folder": folder,
                        "image_id": i,
                        "site_id": order[j],
                        "unit": units[0],
                        "method": "shape run + ordered segmentation",
                    }
                )
    df = pd.DataFrame(rows)
    df["has_plume"] = [
        bool(np.load(root / f"{r.folder}/label/{r.image_id}.npy").sum() > 0)
        for r in df.itertuples()
    ]
    df["ncc"] = [
        float(_zflat(ch4net_crop(root / f"{r.folder}/s2/{r.image_id}.npy")) @ site_tmpl[r.site_id])
        / site_tmpl[r.site_id].size
        for r in df.itertuples()
    ]
    return df.sort_values(["folder", "image_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    """CLI: write data/ch4net_index_all.parquet."""
    import argparse

    from plumescout.config import load_yaml, resolve_path

    p = argparse.ArgumentParser(description="Site ids for all CH4Net images.")
    p.add_argument("--out", default="data/ch4net_index_all.parquet")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_yaml("sites.yaml")
    scenes = pd.read_parquet(resolve_path(cfg["paths"]["scenes_table"]))
    scenes = scenes[pd.to_datetime(scenes.date).dt.year == 2021]
    df = build_index(resolve_path("data/raw/ch4net"), scenes, cfg["sentinel2"]["bands"])
    df["source"] = "CH4Net (av555/ch4net, CC BY-NC-ND 4.0); site ids inferred by image matching"
    out = resolve_path(args.out)
    df.to_parquet(out, index=False)
    logger.info("wrote %s (%d images)", out, len(df))


if __name__ == "__main__":
    main()
