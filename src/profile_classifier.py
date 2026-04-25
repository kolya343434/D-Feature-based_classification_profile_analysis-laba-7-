# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import List, NamedTuple, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\ARIALUNI.TTF",
    "C:\\Windows\\Fonts\\times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]

class SymbolHypothesis(NamedTuple):
    char: str
    closeness: float
    distance: float
    profile_closeness: float = 0.0
    profile_distance: float = 0.0

@dataclass
class RecognitionResult:
    text: str
    hypotheses: List[List[SymbolHypothesis]]
    best_string: str
    errors: int
    accuracy: float
    levenshtein_distance: int
    image: Image.Image

@dataclass
class SymbolPrototype:
    char: str
    features: np.ndarray
    profile: List[int]


def load_font(size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    candidates: Sequence[Optional[str]] = ([font_path] if font_path else []) + FONT_CANDIDATES
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_char_image(char: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    canvas = Image.new("L", (font.size * 2, font.size * 2), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((font.size // 2, font.size // 2), char, fill=0, font=font)
    bbox = canvas.getbbox()
    if not bbox:
        return canvas.crop((0, 0, 1, 1))
    return canvas.crop(bbox)


def render_text_image(text: str, font: ImageFont.FreeTypeFont, spacing: int) -> tuple[Image.Image, List[tuple[int, int, int, int]]]:
    if not text:
        raise ValueError("text must contain at least one character")
    char_images = [render_char_image(ch, font) for ch in text]
    height = max(img.height for img in char_images)
    total_width = sum(img.width for img in char_images) + spacing * (len(char_images) - 1)
    output = Image.new("L", (total_width, height), 255)
    boxes: List[tuple[int, int, int, int]] = []
    x = 0
    for img in char_images:
        y = height - img.height
        output.paste(img, (x, y))
        boxes.append((x, y, x + img.width, y + img.height))
        x += img.width + spacing
    return output, boxes


def compute_features(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("L"), dtype=float)
    arr = 255 - arr
    arr = np.clip(arr, 0, 255)
    arr /= 255.0
    mass = arr.sum()
    height, width = arr.shape
    if mass < 1e-6 or width == 0 or height == 0:
        return np.zeros(5, dtype=float)
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    centroid_x = (arr * x_coords[np.newaxis, :]).sum() / mass
    centroid_y = (arr * y_coords[:, np.newaxis]).sum() / mass
    y_diff = y_coords[:, np.newaxis] - centroid_y
    x_diff = x_coords[np.newaxis, :] - centroid_x
    inertia_x = (arr * y_diff**2).sum()
    inertia_y = (arr * x_diff**2).sum()
    mass_norm = mass / (height * width)
    centroid_x_norm = centroid_x / width
    centroid_y_norm = centroid_y / height
    inertia_x_norm = inertia_x / (mass * height * height)
    inertia_y_norm = inertia_y / (mass * width * width)
    return np.array([
        mass_norm,
        centroid_x_norm,
        centroid_y_norm,
        inertia_x_norm,
        inertia_y_norm,
    ], dtype=float)


def resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(target_len, dtype=float)
    if values.size == target_len:
        return values.astype(float, copy=False)
    x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(x_new, x_old, values.astype(float, copy=False))


def compute_profile_sequence(
    image: Image.Image,
    *,
    bins: int = 32,
    levels: int = 10,
) -> List[int]:
    arr = np.array(image.convert("L"), dtype=float)
    arr = 255 - arr
    arr = np.clip(arr, 0, 255) / 255.0
    height, width = arr.shape
    if height == 0 or width == 0 or arr.sum() < 1e-6:
        return []

    v = arr.sum(axis=0) / max(1.0, float(height))
    h = arr.sum(axis=1) / max(1.0, float(width))
    v = resample_1d(v, bins)
    h = resample_1d(h, bins)

    def quantize(x: np.ndarray) -> List[int]:
        x = np.clip(x, 0.0, 1.0)
        q = np.floor(x * (levels - 1e-9)).astype(int)
        q = np.clip(q, 0, levels - 1)
        return q.tolist()

    return quantize(v) + quantize(h)


def build_prototypes(alphabet: str, font: ImageFont.FreeTypeFont) -> List[SymbolPrototype]:
    prototypes = []
    for char in alphabet:
        char_image = render_char_image(char, font)
        feature_ring = compute_features(char_image)
        profile = compute_profile_sequence(char_image)
        prototypes.append(SymbolPrototype(char, feature_ring, profile))
    return prototypes


def compute_symbol_hypotheses(
    features: np.ndarray,
    symbol_profile: List[int],
    prototypes: List[SymbolPrototype],
    *,
    use_profiles: bool,
    profile_weight: float,
) -> List[SymbolHypothesis]:
    candidates: List[SymbolHypothesis] = []
    for prototype in prototypes:
        distance = float(np.linalg.norm(prototype.features - features))
        closeness = 1.0 / (1.0 + distance)
        if use_profiles:
            profile_distance = float(levenshtein_seq(symbol_profile, prototype.profile))
            norm = float(max(1, max(len(symbol_profile), len(prototype.profile))))
            profile_distance /= norm
            profile_closeness = 1.0 / (1.0 + profile_distance)
            closeness = (1.0 - profile_weight) * closeness + profile_weight * profile_closeness
        else:
            profile_closeness = 0.0
            profile_distance = 0.0
        candidates.append(
            SymbolHypothesis(
                prototype.char,
                float(closeness),
                float(distance),
                float(profile_closeness),
                float(profile_distance),
            )
        )
    candidates.sort(key=lambda hyp: hyp.closeness, reverse=True)
    return candidates


def run_recognition(
    text: str,
    prototypes: List[SymbolPrototype],
    font: ImageFont.FreeTypeFont,
    spacing: int,
    *,
    use_profiles: bool,
    profile_weight: float,
) -> RecognitionResult:
    image, boxes = render_text_image(text, font, spacing)
    hypotheses: List[List[SymbolHypothesis]] = []
    best_chars: List[str] = []
    for box in boxes:
        symbol_image = image.crop(box)
        features = compute_features(symbol_image)
        profile = compute_profile_sequence(symbol_image)
        candidates = compute_symbol_hypotheses(
            features,
            profile,
            prototypes,
            use_profiles=use_profiles,
            profile_weight=profile_weight,
        )
        hypotheses.append(candidates)
        best_chars.append(candidates[0].char if candidates else "?")
    prediction = "".join(best_chars)
    errors = sum(1 for expected, actual in zip(text, prediction) if expected != actual)
    errors += abs(len(text) - len(prediction))
    accuracy = 100.0 * (1.0 - errors / len(text)) if text else 0.0
    accuracy = max(0.0, accuracy)
    lev_distance = levenshtein(prediction, text)
    return RecognitionResult(
        text=text,
        hypotheses=hypotheses,
        best_string=prediction,
        errors=errors,
        accuracy=accuracy,
        levenshtein_distance=lev_distance,
        image=image,
    )


def levenshtein(source: str, target: str) -> int:
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for i, sc in enumerate(source, 1):
        current = [i]
        for j, tc in enumerate(target, 1):
            insert_cost = current[-1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if sc == tc else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def levenshtein_seq(source: Sequence[int], target: Sequence[int]) -> int:
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for i, sc in enumerate(source, 1):
        current = [i]
        for j, tc in enumerate(target, 1):
            insert_cost = current[-1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if sc == tc else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def save_hypotheses(hypotheses: List[List[SymbolHypothesis]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, candidates in enumerate(hypotheses, 1):
            row = ", ".join(f'("{hyp.char}", {hyp.closeness:.3f})' for hyp in candidates[:10])
            handle.write(f"{index}: [{row}]\n")


def save_metrics(base: RecognitionResult, variant: RecognitionResult, base_font: int, variant_font: int, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Распознавание символов и эксперимент\n\n")
        handle.write(f"| Сравнение | Базовый ({base_font} pt) | Вариант ({variant_font} pt) |\n")
        handle.write("|---|---|---|\n")
        handle.write(f"| Предсказанная строка | `{base.best_string}` | `{variant.best_string}` |\n")
        handle.write(f"| Ошибки | {base.errors} | {variant.errors} |\n")
        handle.write(f"| Точность (%) | {base.accuracy:.2f} | {variant.accuracy:.2f} |\n")
        handle.write(f"| Расстояние Левенштейна | {base.levenshtein_distance} | {variant.levenshtein_distance} |\n")
        handle.write("\n> Вывод: изменение размера шрифта влияет на плотность признаков и немного меняет степень уверенности классификатора.\n")


def describe_result(result: RecognitionResult) -> str:
    return (
        f"Грунтовая строка: `{result.text}` -> лучшая гипотеза `{result.best_string}` | "
        f"ошибки {result.errors}, точность {result.accuracy:.2f}%, lev {result.levenshtein_distance}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Изучаем меру близости символов по признакам и сравниваем гипотезы."
    )
    parser.add_argument("--text", "-t", default="система", help="распознаваемая строка")
    parser.add_argument("--alphabet", "-a", default=DEFAULT_ALPHABET, help="набор символов алфавита")
    parser.add_argument("--font-size", "-s", type=int, default=72, help="размер шрифта (pt)")
    parser.add_argument("--variation-font-size", "-v", type=int, default=86, help="размер шрифта для эксперимента")
    parser.add_argument("--font-path", "-f", default="", help="путь к файлу шрифта (опционально)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs"), help="куда сохранять файлы")
    parser.add_argument("--use-profiles", action="store_true", help="добавить сравнение профилей (Левенштейн)")
    parser.add_argument("--profile-weight", type=float, default=0.35, help="вес профиля в итоговой близости")
    args = parser.parse_args()
    base_spacing = max(2, args.font_size // 4)
    variant_spacing = max(2, args.variation_font_size // 4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_font = load_font(args.font_size, args.font_path or None)
    variant_font = load_font(args.variation_font_size, args.font_path or None)
    prototypes = build_prototypes(args.alphabet, base_font)
    base_result = run_recognition(
        args.text,
        prototypes,
        base_font,
        base_spacing,
        use_profiles=args.use_profiles,
        profile_weight=args.profile_weight,
    )
    variant_result = run_recognition(
        args.text,
        prototypes,
        variant_font,
        variant_spacing,
        use_profiles=args.use_profiles,
        profile_weight=args.profile_weight,
    )
    base_png = args.output_dir / "base_render.png"
    variant_png = args.output_dir / "variation_render.png"
    base_result.image.save(base_png)
    variant_result.image.save(variant_png)
    save_hypotheses(base_result.hypotheses, args.output_dir / "hypotheses.txt")
    save_metrics(base_result, variant_result, args.font_size, args.variation_font_size, args.output_dir / "metrics.md")
    print("[основной проход]", describe_result(base_result))
    print("[эксперимент с другим шрифтом]", describe_result(variant_result))
    print(f"Графика сохранена: {base_png}, {variant_png}")
    print(f"Гипотезы -> {args.output_dir / 'hypotheses.txt'}")


if __name__ == "__main__":
    main()
