from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .binarize import binarize, despeckle
from .cc import connected_components, sort_components_left_to_right
from .features import euclidean_distance, extract_features, similarity_from_distance
from .profiles import profile_similarity
from .templates import Templates


@dataclass(frozen=True)
class Hypothesis:
    ch: str
    score: float


def segment_symbols(binary: np.ndarray, *, min_area: int = 30) -> list[np.ndarray]:
    labels, comps = connected_components(binary)
    comps = [c for c in comps if c["area"] >= min_area]
    comps = sort_components_left_to_right(comps)

    symbols: list[np.ndarray] = []
    for c in comps:
        crop = binary[c["y0"] : c["y1"], c["x0"] : c["x1"]]
        symbols.append(crop)
    return symbols


def recognize_line(
    img: Image.Image,
    templates: Templates,
    *,
    min_area: int = 30,
    use_profiles: bool = False,
    profile_weight: float = 0.35,
) -> tuple[list[list[Hypothesis]], str]:
    gray = np.array(img.convert("L"), dtype=np.uint8)
    binary = binarize(gray, invert=False)
    binary = despeckle(binary, min_area=8)
    symbols = segment_symbols(binary, min_area=min_area)

    all_hyp: list[list[Hypothesis]] = []
    best_chars: list[str] = []

    for sym in symbols:
        f = extract_features(sym)
        scores: list[Hypothesis] = []
        for ch, tf, tb in zip(templates.alphabet, templates.features, templates.binaries or [None] * len(templates.alphabet)):
            d = euclidean_distance(f, tf)
            s = similarity_from_distance(d)
            if use_profiles and tb is not None:
                ps = profile_similarity(sym, tb)
                s = (1.0 - profile_weight) * s + profile_weight * ps
            scores.append(Hypothesis(ch=ch, score=float(s)))

        scores.sort(key=lambda h: h.score, reverse=True)
        all_hyp.append(scores)
        best_chars.append(scores[0].ch if scores else "?")

    return all_hyp, "".join(best_chars)


def error_stats(pred: str, truth: str) -> tuple[int, float]:
    m = min(len(pred), len(truth))
    errors = sum(1 for i in range(m) if pred[i] != truth[i]) + abs(len(pred) - len(truth))
    acc = 0.0 if len(truth) == 0 else 100.0 * (len(truth) - errors) / len(truth)
    return errors, acc

