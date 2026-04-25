from __future__ import annotations

import numpy as np


def projection_profiles(binary_symbol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = binary_symbol.shape
    if h == 0 or w == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    v = binary_symbol.sum(axis=0).astype(np.float64) / max(1.0, float(h))
    hprof = binary_symbol.sum(axis=1).astype(np.float64) / max(1.0, float(w))
    return v, hprof


def quantize_profile(profile: np.ndarray, *, bins: int = 10) -> list[int]:
    if profile.size == 0:
        return []
    p = np.clip(profile, 0.0, 1.0)
    q = np.floor(p * (bins - 1e-9)).astype(int)
    q = np.clip(q, 0, bins - 1)
    return q.tolist()


def levenshtein(a: list[int], b: list[int]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)

    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, cur = cur, prev
    return prev[-1]


def profile_similarity(
    binary_symbol_a: np.ndarray,
    binary_symbol_b: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    av, ah = projection_profiles(binary_symbol_a)
    bv, bh = projection_profiles(binary_symbol_b)
    aseq = quantize_profile(np.concatenate([av, ah]), bins=bins)
    bseq = quantize_profile(np.concatenate([bv, bh]), bins=bins)
    if not aseq and not bseq:
        return 1.0
    d = levenshtein(aseq, bseq)
    norm = max(1, max(len(aseq), len(bseq)))
    return 1.0 / (1.0 + d / norm)

