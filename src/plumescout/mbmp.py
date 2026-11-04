"""Multi-band methane enhancement (Varon et al. 2021, AMT 14, 2771).

Inputs are Sentinel-2 L1C top-of-atmosphere reflectance arrays (unitless) for
B11 (~1610 nm) and B12 (~2190 nm) on the same 2D pixel grid. Outputs are
fractional enhancement arrays dR on that grid (unitless, no CRS change).

Sign convention (Varon et al. 2021): dR = (c * R12 - R11) / R11. Methane absorbs
in B12 and not much in B11, so plume pixels have dR < 0.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Robust sigma from median absolute deviation for a normal distribution.
_MAD_TO_SIGMA = 1.4826


def _check_shapes(*arrays: np.ndarray) -> None:
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All bands must share one shape, got {sorted(shapes)}")


def _valid(r11: np.ndarray, r12: np.ndarray) -> np.ndarray:
    """Pixels usable for MBMP: finite and strictly positive in both bands."""
    return np.isfinite(r11) & np.isfinite(r12) & (r11 > 0) & (r12 > 0)


def fit_c(r11: np.ndarray, r12: np.ndarray) -> float:
    """Least-squares scale c minimising ||R11 - c * R12||^2 over valid pixels.

    Args:
        r11, r12: B11 and B12 TOA reflectance, same 2D shape.
    Returns:
        c (unitless). Raises ValueError if no pixel is valid.
    """
    r11 = np.asarray(r11, dtype=np.float64)
    r12 = np.asarray(r12, dtype=np.float64)
    _check_shapes(r11, r12)
    ok = _valid(r11, r12)
    if not ok.any():
        raise ValueError("No valid pixels to fit c (need finite, positive B11 and B12)")
    a, b = r11[ok], r12[ok]
    return float(np.dot(a, b) / np.dot(b, b))


def single_pass(r11: np.ndarray, r12: np.ndarray, c: float | None = None) -> np.ndarray:
    """Multi-band single-pass enhancement dR = (c * R12 - R11) / R11.

    Args:
        r11, r12: B11 and B12 TOA reflectance, same 2D shape.
        c: scale factor; fitted with `fit_c` over the chip if None.
    Returns:
        float64 array, same shape; NaN where a pixel is invalid. Plumes are negative.
    """
    r11 = np.asarray(r11, dtype=np.float64)
    r12 = np.asarray(r12, dtype=np.float64)
    _check_shapes(r11, r12)
    if c is None:
        c = fit_c(r11, r12)
    ok = _valid(r11, r12)
    dr = np.full(r11.shape, np.nan)
    dr[ok] = (c * r12[ok] - r11[ok]) / r11[ok]
    return dr


def multi_pass(
    r11: np.ndarray,
    r12: np.ndarray,
    ref_r11: np.ndarray,
    ref_r12: np.ndarray,
) -> np.ndarray:
    """Multi-band multi-pass enhancement: target dR minus reference-scene dR.

    Each scene gets its own fitted c. The reference should be a recent clear
    scene of the same chip (same grid), so static surface features cancel.

    Returns:
        float64 array, same shape; NaN where either scene is invalid. Plumes are negative.
    """
    _check_shapes(np.asarray(r11), np.asarray(r12), np.asarray(ref_r11), np.asarray(ref_r12))
    return single_pass(r11, r12) - single_pass(ref_r11, ref_r12)


def threshold_mask(dr: np.ndarray, k: float) -> np.ndarray:
    """Binary plume mask: pixels more than k robust sigmas below the chip median.

    Robust sigma = 1.4826 * MAD over finite pixels. Only negative anomalies count
    (methane lowers dR). NaN pixels are never flagged.

    Args:
        dr: enhancement array from `single_pass` or `multi_pass`.
        k: threshold in robust standard deviations (set in config, not tuned on Gulf data).
    Returns:
        bool array, same shape.
    """
    dr = np.asarray(dr, dtype=np.float64)
    finite = np.isfinite(dr)
    mask = np.zeros(dr.shape, dtype=bool)
    if not finite.any():
        return mask
    vals = dr[finite]
    med = float(np.median(vals))
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(vals - med)))
    if sigma == 0.0:
        logger.debug("Zero spread in dR; flagging only pixels strictly below the median")
    thr = med - k * sigma
    mask[finite] = dr[finite] < thr
    return mask
