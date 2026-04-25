from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .binarize import binarize
from .features import extract_features
from .render import render_text_line, save_image


@dataclass(frozen=True)
class Templates:
    alphabet: list[str]
    features: np.ndarray  # (N, D)
    binaries: list[np.ndarray]  # per-character binarized symbol crops

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            alphabet=np.array(self.alphabet),
            features=self.features,
        )

    @staticmethod
    def load(path: str | Path) -> "Templates":
        data = np.load(Path(path), allow_pickle=False)
        alphabet = [str(x) for x in data["alphabet"].tolist()]
        features = data["features"].astype(np.float64, copy=False)
        # binaries are optional (not stored in npz)
        return Templates(alphabet=alphabet, features=features, binaries=[])


def build_templates(
    alphabet: str,
    *,
    font_path: str | None = None,
    font_size: int = 48,
    out_dir: str | Path | None = None,
) -> Templates:
    chars = [c for c in alphabet]
    feats: list[np.ndarray] = []
    bins: list[np.ndarray] = []

    for c in chars:
        img = render_text_line(c, font_path=font_path, font_size=font_size, padding=8)
        gray = np.array(img, dtype=np.uint8)
        fg = binarize(gray, invert=False)

        ys, xs = np.nonzero(fg)
        if len(xs) == 0:
            crop = fg
        else:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            crop = fg[y0:y1, x0:x1]

        feats.append(extract_features(crop))
        bins.append(crop)

        if out_dir is not None:
            out_dir_p = Path(out_dir)
            save_image(img, out_dir_p / f"tpl_{ord(c):04x}.png")
            np.save(out_dir_p / f"bin_{ord(c):04x}.npy", crop.astype(np.uint8))

    features = np.vstack(feats) if feats else np.zeros((0, 5), dtype=np.float64)
    return Templates(alphabet=chars, features=features, binaries=bins)
