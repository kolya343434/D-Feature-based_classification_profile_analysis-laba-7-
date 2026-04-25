# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

sys.path.insert(0, str(ROOT))

from feature_classifier.recognize import recognize_line  # noqa: E402
from feature_classifier.render import render_text_line, save_image  # noqa: E402
from feature_classifier.templates import build_templates  # noqa: E402


def http_get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def http_download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        out_path.write_bytes(r.read())


def download_slavcorpora_examples(*, manuscript_id: str, n: int = 3) -> list[Path]:
    images = http_get_json(f"https://www.slavcorpora.ru/api/manuscripts/{manuscript_id}/images")
    if not isinstance(images, list):
        raise RuntimeError("Unexpected API response for manuscript images")
    paths: list[Path] = []
    for i, item in enumerate(images[:n], start=1):
        filename = item["filename"]
        url = f"https://www.slavcorpora.ru/images/{filename}"
        out = ASSETS / f"slavcorpora_{i}.jpeg"
        http_download(url, out)
        paths.append(out)
    return paths


def _to_tile(binary: np.ndarray, *, size: int = 64) -> Image.Image:
    img = Image.fromarray((~binary).astype(np.uint8) * 255)  # black fg
    img = img.resize((size, size), Image.Resampling.NEAREST)
    return img.convert("RGB")


def make_segments_grid(
    symbols: list[np.ndarray],
    labels: list[str],
    *,
    tile: int = 72,
    pad: int = 10,
) -> Image.Image:
    cols = min(10, max(1, len(symbols)))
    rows = (len(symbols) + cols - 1) // cols
    w = cols * tile + (cols + 1) * pad
    h = rows * tile + (rows + 1) * pad + 28
    out = Image.new("RGB", (w, h), color=(245, 246, 250))
    d = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype(str(Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / "arial.ttf"), 16)
    except Exception:
        font = ImageFont.load_default()

    d.text((pad, 6), "Сегментация + лучшая гипотеза", fill=(20, 20, 20), font=font)
    y0 = 28
    for idx, (sym, ch) in enumerate(zip(symbols, labels)):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile + pad)
        y = y0 + pad + r * (tile + pad)
        tile_img = _to_tile(sym, size=tile)
        out.paste(tile_img, (x, y))
        d.text((x + 4, y + 4), ch, fill=(200, 30, 30), font=font)
    return out


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # 1) Slavcorpora examples (for README illustrations)
    manuscript_id = "856066a1-8663-4e31-9fbf-b740ab965c8c"
    download_slavcorpora_examples(manuscript_id=manuscript_id, n=3)

    # 2) Generate our own clean line + recognition demo (the lab pipeline)
    alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    text = "приветмир"
    font_path = str(Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / "arial.ttf")
    tpl = build_templates(alphabet, font_path=font_path, font_size=52)

    base = render_text_line(text, font_path=font_path, font_size=52, padding=16)
    var = render_text_line(text, font_path=font_path, font_size=56, padding=16)
    save_image(base.convert("RGB"), ASSETS / "generated_line_base.png")
    save_image(var.convert("RGB"), ASSETS / "generated_line_variant.png")

    all_hyp, best = recognize_line(base, tpl, min_area=30)
    with (ASSETS / "demo_hypotheses.txt").open("w", encoding="utf-8") as f:
        for i, hyp in enumerate(all_hyp, start=1):
            pairs = [(h.ch, round(float(h.score), 4)) for h in hyp[:8]]
            f.write(f"{i}: {pairs}\n")

    # Re-run segmentation to visualize symbol crops
    from feature_classifier.binarize import binarize, despeckle  # noqa: E402
    from feature_classifier.recognize import segment_symbols  # noqa: E402

    gray = np.array(base.convert("L"), dtype=np.uint8)
    binary = despeckle(binarize(gray, invert=False), min_area=8)
    syms = segment_symbols(binary, min_area=30)
    best_labels = [h[0].ch for h in all_hyp]
    grid = make_segments_grid(syms, best_labels)
    save_image(grid, ASSETS / "segments_grid.png")

    # 3) Experiment summary JSON
    all_hyp2, best2 = recognize_line(var, tpl, min_area=30)
    summary = {"base": {"truth": text, "best": best}, "variant": {"truth": text, "best": best2}}
    (ASSETS / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Generated README assets in:", ASSETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
