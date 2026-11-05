"""CH4Net dataset for training (option (a): CH4Net's own arrays).

CH4Net release format (found by inspection, 2026-11-05):
- `s2/{id}.npy`: (H, W, 12) uint8 display values for B1..B8, B8A, B9, B11, B12 (no B10).
  These are not physical reflectance; this is a known deviation from hard rule 6's intent,
  accepted for reproducing the paper.
- `mbmp/{id}.npy`: (H, W, 4) RGBA rendering of the MBMP field with matplotlib `magma`.
  Decoded back to a scalar in [0, 1] by nearest palette colour.
- `label/{id}.npy`: (H, W) float hand-drawn plume mask (> 0 = plume).

Model input: the configured bands (uint8 / 255) + decoded MBMP, e.g. 12 bands + MBMP = 13.
Augmentation: flips and 90-degree rotations only (hard rule 6).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CH4NET_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


@lru_cache(maxsize=1)
def _magma_tree():
    import matplotlib
    from scipy.spatial import cKDTree

    lut = matplotlib.colormaps["magma"](np.arange(256) / 255)[:, :3] * 255
    return cKDTree(lut)


def decode_magma(rgba: np.ndarray) -> np.ndarray:
    """(H, W, 3 or 4) uint8 magma rendering -> (H, W) float32 scalar in [0, 1]."""
    rgb = np.asarray(rgba)[..., :3].reshape(-1, 3).astype(np.float64)
    _, idx = _magma_tree().query(rgb)
    return (idx / 255).astype(np.float32).reshape(rgba.shape[:2])


def load_ch4net(
    root: Path, folder: str, image_id: int, bands: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return x (C, H, W) float32 and y (H, W) float32 for one CH4Net image."""
    s2 = np.load(root / folder / "s2" / f"{image_id}.npy")
    idx = [CH4NET_BANDS.index(b) for b in bands]
    x = np.moveaxis(s2[..., idx], -1, 0).astype(np.float32) / 255.0
    m = decode_magma(np.load(root / folder / "mbmp" / f"{image_id}.npy"))
    x = np.concatenate([x, m[None]], axis=0)
    y = (np.load(root / folder / "label" / f"{image_id}.npy") > 0).astype(np.float32)
    return x, y


def augment(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal / vertical flip and 90-degree rotation, applied identically to x and y."""
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if rng.random() < 0.5:
        x, y = x[:, ::-1, :], y[::-1, :]
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def random_crop(
    x: np.ndarray, y: np.ndarray, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Square crop of `size` px at a random position."""
    h, w = y.shape
    r = int(rng.integers(0, h - size + 1))
    c = int(rng.integers(0, w - size + 1))
    return x[:, r : r + size, c : c + size], y[r : r + size, c : c + size]


def pad_to(x: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad (C, H0, W0) at bottom/right to (C, h, w). Returns padded x and valid mask."""
    c, h0, w0 = x.shape
    out = np.zeros((c, h, w), dtype=x.dtype)
    out[:, :h0, :w0] = x
    valid = np.zeros((h, w), dtype=bool)
    valid[:h0, :w0] = True
    return out, valid


class CH4NetDataset(Dataset):
    """Rows of `items` need `folder` and `image_id`. Training mode crops and augments;
    eval mode pads to (pad_h, pad_w) and also returns the valid-pixel mask."""

    def __init__(
        self,
        root: Path,
        items: pd.DataFrame,
        bands: list[str],
        train: bool,
        crop: int = 160,
        pad_hw: tuple[int, int] = (256, 192),
        seed: int = 0,
    ) -> None:
        self.root, self.bands, self.train = Path(root), list(bands), train
        self.items = items[["folder", "image_id"]].reset_index(drop=True)
        self.crop, self.pad_hw, self.seed = crop, pad_hw, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        r = self.items.iloc[i]
        x, y = load_ch4net(self.root, r.folder, int(r.image_id), self.bands)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            x, y = random_crop(x, y, self.crop, rng)
            x, y = augment(x, y, rng)
            return torch.from_numpy(x), torch.from_numpy(y[None])
        xp, valid = pad_to(x, *self.pad_hw)
        yp, _ = pad_to(y[None], *self.pad_hw)
        return torch.from_numpy(xp), torch.from_numpy(yp), torch.from_numpy(valid), i
