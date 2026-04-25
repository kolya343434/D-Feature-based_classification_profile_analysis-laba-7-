from __future__ import annotations

import numpy as np


def otsu_threshold(gray: np.ndarray) -> int:
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8, copy=False)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total == 0:
        return 128

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    t = int(np.nanargmax(sigma_b2))
    return t


def binarize(gray: np.ndarray, *, invert: bool | None = None) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8, copy=False)

    t = otsu_threshold(gray)
    if invert is None:
        # Most documents are dark ink on light paper.
        invert = bool(gray.mean() < 127)

    fg = gray > t if invert else gray < t
    return fg.astype(bool)


def despeckle(binary: np.ndarray, *, min_area: int = 8) -> np.ndarray:
    from .cc import connected_components

    labels, comps = connected_components(binary)
    keep = np.zeros(len(comps) + 1, dtype=bool)  # 0 is background
    keep[0] = False
    for idx, comp in enumerate(comps, start=1):
        keep[idx] = comp["area"] >= min_area
    return keep[labels]

