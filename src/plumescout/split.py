"""Train / val / test splits for CH4Net images.

Hard rule 3: split by site, never by image. Sites closer than `eps_m` are grouped
(DBSCAN in metres) and whole groups go to one side.

- `site_year_split`: primary. Test = 2021 (CH4Net `test` folder) images from test groups;
  train and val = 2017 to 2020 (`train` and `val` folders) images from the other groups.
- `paper_split`: CH4Net's own folders (by year only, same sites in train and test). Used
  only to compare with the paper's 84% recall; flagged `reproduction_only`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plumescout.sites import cluster_points


def site_groups(sites: pd.DataFrame, eps_m: float) -> dict[str, int]:
    """Map site_id -> group id, grouping sites within eps_m metres (EPSG:4326 input)."""
    g = cluster_points(sites.lon.to_numpy(), sites.lat.to_numpy(), eps_m=eps_m)
    return dict(zip(sites.site_id, (int(x) for x in g), strict=True))


def assert_site_disjoint(*site_sets: set[str]) -> None:
    """Raise AssertionError if any site appears in more than one set."""
    seen: set[str] = set()
    for s in site_sets:
        overlap = seen & s
        assert not overlap, f"Sites in more than one split: {sorted(overlap)}"
        seen |= s


def site_year_split(
    images: pd.DataFrame,
    groups: dict[str, int],
    test_frac: float,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Assign each image to train / val / test / unused by site group and year folder.

    Args:
        images: rows with `site_id` and `folder` in {train, val, test} (CH4Net folders:
            train and val hold 2017 to 2020, test holds 2021).
        groups: site_id -> group id from `site_groups`.
        test_frac, val_frac: share of groups sent to test and val (at least one each).
    Returns:
        Copy of `images` with a `split` column. Rows that would cross the year rule
        (2021 images of train groups, 2017 to 2020 images of test groups) are `unused`.
    """
    ids = sorted(set(groups.values()))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(ids))
    n_test = max(1, round(test_frac * len(ids)))
    n_val = max(1, round(val_frac * len(ids)))
    test_g, val_g = set(order[:n_test]), set(order[n_test : n_test + n_val])

    g = images.site_id.map(groups)
    if g.isna().any():
        missing = sorted(images.site_id[g.isna()].unique())
        raise KeyError(f"Sites without a group: {missing}")
    early = images.folder.isin(["train", "val"])
    out = images.copy()
    out["split"] = "unused"
    out.loc[g.isin(test_g) & ~early, "split"] = "test"
    out.loc[g.isin(val_g) & early, "split"] = "val"
    out.loc[~g.isin(test_g | val_g) & early, "split"] = "train"
    assert_site_disjoint(
        *(set(out.loc[out.split == k, "site_id"]) for k in ("train", "val", "test"))
    )
    return out


def paper_split(images: pd.DataFrame) -> pd.DataFrame:
    """CH4Net's own split (folder = split). Not site-disjoint: reproduction only."""
    out = images.copy()
    out["split"] = out.folder
    out.attrs["reproduction_only"] = True
    return out
