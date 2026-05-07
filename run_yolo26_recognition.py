#!/usr/bin/env python3
"""
Generate recognition_yolo26.txt for each dataset folder using Ultralytics YOLO.

For each folder data/dataset/<id>/ that contains studSolution.txt:
- process page_*.jpg in numeric order
- run YOLO26 recognition on preprocessed images
- write one line per image to recognition_yolo26.txt
  (letters separated by single spaces, fallback '?')
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


PAGE_RE = re.compile(r"^page_(\d+)\.jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate recognition_yolo26.txt files per dataset.")
    p.add_argument(
        "--dataset-root",
        default="data/dataset",
        help="Root folder containing numeric dataset folders.",
    )
    p.add_argument(
        "--model-path",
        required=True,
        help="YOLO26 model checkpoint path (.pt).",
    )
    p.add_argument(
        "--output-name",
        default="recognition_yolo26.txt",
        help="Output filename per dataset folder.",
    )
    p.add_argument(
        "--fail-token",
        default="?",
        help="Token to use when no letters are recognized or on page-level error.",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--max-det", type=int, default=100, help="Maximum detections per image.")
    p.add_argument("--device", default="", help="Inference device, e.g. '', 'cpu', 'mps', '0'.")
    p.add_argument("--debug-log", default=None, help="Optional debug log file for per-page class diagnostics.")
    return p.parse_args()


def iter_dataset_dirs(dataset_root: Path) -> Iterable[Path]:
    dirs = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()]
    for d in sorted(dirs, key=lambda x: int(x.name)):
        yield d


def list_page_images(dataset_dir: Path) -> List[Path]:
    pages: List[Tuple[int, Path]] = []
    for path in dataset_dir.iterdir():
        m = PAGE_RE.match(path.name)
        if not m:
            continue
        pages.append((int(m.group(1)), path))
    pages.sort(key=lambda t: t[0])
    return [p for _, p in pages]


def preprocess_like_eklausur(src_path: Path, dst_path: Path) -> None:
    """
    Equivalent to eKlausur run_tests.copy_invert_blurr(...):
    grayscale -> Gaussian blur (k=9 if <256 colors else k=7) -> invert -> save jpg.
    """
    img = Image.open(src_path).convert("L")
    img_np = np.array(img)
    colors_num = np.unique(img_np).shape[0]
    blur_kernel = 9 if colors_num < 256 else 7
    filtered = cv2.GaussianBlur(img_np, (blur_kernel, blur_kernel), 0)
    inv = 255 - filtered
    Image.fromarray(inv).save(dst_path, quality=100, subsampling=0)


def normalize_letter(raw: str, fail_token: str) -> str:
    if not raw:
        return fail_token
    if raw == "?":
        return "?"
    if raw == "x":
        return fail_token
    if len(raw) == 1 and raw.isalpha():
        return raw.upper()
    return fail_token


def parse_digit_label(raw: str) -> int | None:
    if len(raw) == 2 and raw[0] == "D" and raw[1].isdigit():
        d = int(raw[1])
        if 1 <= d <= 9:
            return d
    if len(raw) == 1 and raw.isdigit():
        d = int(raw)
        if 1 <= d <= 9:
            return d
    return None


def letters_from_prediction(result, fail_token: str) -> Tuple[List[str], Dict[str, int]]:
    names = result.names
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return [fail_token], {}

    class_histo: Dict[str, int] = {}
    letters: List[Dict[str, float | str]] = []
    digits_all: List[Dict[str, float | int]] = []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        label_raw = str(names.get(int(cls[i]), "")) if isinstance(names, dict) else str(names[int(cls[i])])
        class_histo[label_raw] = class_histo.get(label_raw, 0) + 1
        digit = parse_digit_label(label_raw)
        if digit is not None:
            digits_all.append(
                {"digit": digit, "x1": float(x1), "x2": float(x2), "y1": float(y1), "y2": float(y2), "conf": float(conf[i])}
            )
            continue
        token = normalize_letter(label_raw, fail_token)
        letters.append(
            {"letter": token, "x1": float(x1), "x2": float(x2), "y1": float(y1), "y2": float(y2), "conf": float(conf[i])}
        )

    if not digits_all:
        return [fail_token], class_histo

    # Keep best detection per digit (D1..D9).
    digits_best: Dict[int, Dict[str, float | int]] = {}
    for d in digits_all:
        k = int(d["digit"])
        if k not in digits_best or float(d["conf"]) > float(digits_best[k]["conf"]):
            digits_best[k] = d

    out: List[str] = []
    for digit in sorted(digits_best):
        d = digits_best[digit]
        width = float(d["x2"]) - float(d["x1"])
        cx = float(d["x1"]) + width / 2.0
        candidates: List[Dict[str, float | str]] = []
        for letter in letters:
            lx = float(letter["x1"]) + (float(letter["x2"]) - float(letter["x1"])) / 2.0
            if abs(lx - cx) < width * 2:
                candidates.append(letter)
        if not candidates:
            out.append(fail_token)
            continue
        best = max(candidates, key=lambda x: float(x["conf"]))
        out.append(str(best["letter"]))

    if not out:
        out = [fail_token]
    return out, class_histo


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    print(f"[INFO] Using YOLO26 model: {model_path}")

    processed_datasets = 0
    processed_pages = 0
    failed_pages = 0
    debug_lines: List[str] = []

    with tempfile.TemporaryDirectory(prefix="yolo26_preproc_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for dataset_dir in iter_dataset_dirs(dataset_root):
            stud_solution = dataset_dir / "studSolution.txt"
            if not stud_solution.exists():
                continue

            page_images = list_page_images(dataset_dir)
            if not page_images:
                continue

            out_lines: List[str] = []
            for page_path in page_images:
                try:
                    prep_path = tmp_dir / f"{dataset_dir.name}_{page_path.name}"
                    preprocess_like_eklausur(page_path, prep_path)
                    preds = model.predict(
                        source=str(prep_path),
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        max_det=args.max_det,
                        device=args.device if args.device else None,
                        verbose=False,
                    )
                    letters, class_histo = letters_from_prediction(preds[0], args.fail_token)
                    out_lines.append(" ".join(letters))
                    if args.debug_log:
                        class_str = ", ".join([f"{k}:{v}" for k, v in sorted(class_histo.items())])
                        debug_lines.append(f"{dataset_dir.name}/{page_path.name}\t{class_str}\t{' '.join(letters)}")
                except Exception as ex:
                    failed_pages += 1
                    out_lines.append(args.fail_token)
                    print(f"[WARN] {dataset_dir.name}/{page_path.name}: {ex}")
                    if args.debug_log:
                        debug_lines.append(f"{dataset_dir.name}/{page_path.name}\tERROR\t{ex}")
                processed_pages += 1

            out_file = dataset_dir / args.output_name
            out_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            processed_datasets += 1
            print(f"[OK] {dataset_dir.name}: wrote {out_file.name} ({len(out_lines)} lines)")

    if args.debug_log:
        dbg = Path(args.debug_log).expanduser().resolve()
        dbg.parent.mkdir(parents=True, exist_ok=True)
        dbg.write_text("\n".join(debug_lines) + ("\n" if debug_lines else ""), encoding="utf-8")
        print(f"[INFO] Debug log written: {dbg}")

    print("=== Summary ===")
    print(f"Datasets processed: {processed_datasets}")
    print(f"Pages processed: {processed_pages}")
    print(f"Pages failed (fallback '{args.fail_token}'): {failed_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
