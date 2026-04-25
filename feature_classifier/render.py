from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render_text_line(
    text: str,
    *,
    font_path: str | None = None,
    font_size: int = 48,
    padding: int = 12,
    bg: int = 255,
    fg: int = 0,
) -> Image.Image:
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)

    tmp = Image.new("L", (10, 10), color=bg)
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = (bbox[2] - bbox[0]) + 2 * padding
    h = (bbox[3] - bbox[1]) + 2 * padding

    img = Image.new("L", (w, h), color=bg)
    d = ImageDraw.Draw(img)
    d.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=fg)
    return img


def save_image(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

