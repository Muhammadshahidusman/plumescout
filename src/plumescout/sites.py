"""Site selection helpers (TRD 5.1).

Coordinates are EPSG:4326 (lon, lat in degrees); distances are metres (haversine).
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def cluster_points(lon: np.ndarray, lat: np.ndarray, eps_m: float) -> np.ndarray:
    """DBSCAN (min_samples=1) on great-circle distance; points chained within eps_m share a label.

    Returns integer labels, one per point.
    """
    from sklearn.cluster import DBSCAN

    coords = np.radians(np.column_stack([lat, lon]))
    db = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=1, metric="haversine")
    return db.fit_predict(coords)
