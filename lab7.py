from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from feature_classifier.recognize import error_stats, recognize_line
from feature_classifier.render import render_text_line, save_image
from feature_classifier.templates import Templates, build_templates


DEFAULT_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def _write_hypotheses(path: Path, all_hyp: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, hyp in enumerate(all_hyp, start=1):
            pairs = [(h.ch, round(float(h.score), 4)) for h in hyp]
            f.write(f"{i}: {pairs}\n")


def cmd_render(args: argparse.Namespace) -> int:
    img = render_text_line(
        args.text,
        font_path=args.font,
        font_size=args.size,
        padding=args.padding,
    )
    save_image(img, args.out)
    return 0


def cmd_build_templates(args: argparse.Namespace) -> int:
    tpl = build_templates(
        args.alphabet,
        font_path=args.font,
        font_size=args.size,
        out_dir=args.out_dir,
    )
    tpl.save(args.out)
    print(f"Saved templates: {args.out} (N={len(tpl.alphabet)}, D={tpl.features.shape[1]})")
    return 0


def cmd_recognize(args: argparse.Namespace) -> int:
    img = Image.open(args.image)
    tpl = Templates.load(args.templates)

    # Optional: load per-char binaries for profile similarity if user provides them.
    if args.templates_bin_dir:
        bins: list[np.ndarray] = []
        for ch in tpl.alphabet:
            p = Path(args.templates_bin_dir) / f"bin_{ord(ch):04x}.npy"
            bins.append(np.load(p))
        tpl = Templates(alphabet=tpl.alphabet, features=tpl.features, binaries=bins)

    all_hyp, best = recognize_line(
        img,
        tpl,
        min_area=args.min_area,
        use_profiles=args.use_profiles,
        profile_weight=args.profile_weight,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_hypotheses(outdir / "hypotheses.txt", all_hyp)

    report = {
        "best": best,
        "truth": args.truth,
    }
    if args.truth is not None:
        errors, acc = error_stats(best, args.truth)
        report["errors"] = errors
        report["accuracy_percent"] = round(acc, 2)

    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Best: {best}")
    if args.truth is not None:
        print(f"Truth: {args.truth}")
        print(f"Errors: {report['errors']}, accuracy={report['accuracy_percent']}%")
    print(f"Wrote: {outdir / 'hypotheses.txt'}")
    print(f"Wrote: {outdir / 'report.json'}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    base_size = args.size
    delta = args.delta
    text = args.text
    alphabet = args.alphabet

    tpl = build_templates(alphabet, font_path=args.font, font_size=base_size)

    def run(size: int, tag: str) -> dict:
        img = render_text_line(text, font_path=args.font, font_size=size, padding=14)
        outdir = Path(args.outdir) / tag
        outdir.mkdir(parents=True, exist_ok=True)
        save_image(img, outdir / "line.png")
        all_hyp, best = recognize_line(
            img,
            tpl,
            min_area=args.min_area,
            use_profiles=args.use_profiles,
            profile_weight=args.profile_weight,
        )
        _write_hypotheses(outdir / "hypotheses.txt", all_hyp)
        errors, acc = error_stats(best, text)
        rep = {
            "font_size": size,
            "best": best,
            "truth": text,
            "errors": errors,
            "accuracy_percent": round(acc, 2),
        }
        (outdir / "report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return rep

    rep_a = run(base_size, "base")
    rep_b = run(base_size + delta, f"size_{base_size+delta}")

    summary = {"base": rep_a, "variant": rep_b}
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ЛР7: классификация символов по признакам + анализ профилей."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("render", help="Сгенерировать изображение строки (PIL).")
    pr.add_argument("--text", required=True)
    pr.add_argument("--font", default=None, help="Путь к .ttf/.otf (опционально).")
    pr.add_argument("--size", type=int, default=48)
    pr.add_argument("--padding", type=int, default=14)
    pr.add_argument("--out", default="output/line.png")
    pr.set_defaults(fn=cmd_render)

    pt = sub.add_parser("build-templates", help="Сгенерировать шаблоны алфавита.")
    pt.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    pt.add_argument("--font", default=None)
    pt.add_argument("--size", type=int, default=48)
    pt.add_argument("--out", default="templates/templates.npz")
    pt.add_argument("--out-dir", default=None, help="Если задано, сохраняет PNG-ы символов.")
    pt.set_defaults(fn=cmd_build_templates)

    pc = sub.add_parser("recognize", help="Распознать строку по шаблонам.")
    pc.add_argument("--image", required=True)
    pc.add_argument("--templates", default="templates/templates.npz")
    pc.add_argument("--templates-bin-dir", default=None, help="Директория с bin_XXXX.npy для профилей.")
    pc.add_argument("--outdir", default="output/run")
    pc.add_argument("--truth", default=None, help="Истинная строка (для подсчёта ошибок).")
    pc.add_argument("--min-area", type=int, default=30)
    pc.add_argument("--use-profiles", action="store_true")
    pc.add_argument("--profile-weight", type=float, default=0.35)
    pc.set_defaults(fn=cmd_recognize)

    pe = sub.add_parser("experiment", help="Эксперимент с изменением размера шрифта.")
    pe.add_argument("--text", required=True)
    pe.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    pe.add_argument("--font", default=None)
    pe.add_argument("--size", type=int, default=48)
    pe.add_argument("--delta", type=int, default=4)
    pe.add_argument("--outdir", default="output/experiment")
    pe.add_argument("--min-area", type=int, default=30)
    pe.add_argument("--use-profiles", action="store_true")
    pe.add_argument("--profile-weight", type=float, default=0.35)
    pe.set_defaults(fn=cmd_experiment)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

