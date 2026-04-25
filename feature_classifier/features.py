from __future__ import annotations

import numpy as np


def extract_features(binary_symbol: np.ndarray) -> np.ndarray:
    """
    Normalized feature vector:
      - mass: sum(fg) / (w*h)
      - centroid x,y normalized to [0..1] within bounding box
      - axial moments Ixx, Iyy normalized by mass and bbox size
    """
    h, w = binary_symbol.shape
    if h == 0 or w == 0:
        return np.zeros(5, dtype=np.float64)

    ys, xs = np.nonzero(binary_symbol)
    mass = float(len(xs))
    if mass == 0:
        return np.zeros(5, dtype=np.float64)

    cx = float(xs.mean())
    cy = float(ys.mean())

    mass_n = mass / float(w * h)
    cx_n = 0.0 if w <= 1 else cx / float(w - 1)
    cy_n = 0.0 if h <= 1 else cy / float(h - 1)

    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    ixx = float(np.sum(dy * dy))
    iyy = float(np.sum(dx * dx))

    denom_x = mass * max(1.0, float((h - 1) ** 2))
    denom_y = mass * max(1.0, float((w - 1) ** 2))
    ixx_n = ixx / denom_x
    iyy_n = iyy / denom_y

    return np.array([mass_n, cx_n, cy_n, ixx_n, iyy_n], dtype=np.float64)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def similarity_from_distance(d: float) -> float:
    # d=0 -> 1.0; monotonic decay
    return 1.0 / (1.0 + float(d))

