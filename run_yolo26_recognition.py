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
from typing import Iterable, List, Tuple

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
    p.add_argument(
        "--line-y-ratio",
        type=float,
        default=0.7,
        help="Row grouping tolerance as ratio of median box height for reading-order sort.",
    )
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


def normalize_label(raw: str, fail_token: str) -> str:
    if not raw:
        return fail_token
    if raw == "?":
        return "?"
    if len(raw) == 1 and raw.isalpha():
        return raw.upper()
    return fail_token


def sort_boxes_reading_order(boxes: List[Tuple[float, float, float, str]], line_y_ratio: float) -> List[Tuple[float, float, float, str]]:
    if not boxes:
        return boxes
    heights = sorted([h for _, _, h, _ in boxes])
    median_h = heights[len(heights) // 2]
    y_tol = max(1.0, median_h * line_y_ratio)

    rows: List[List[Tuple[float, float, float, str]]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        x, y, h, token = box
        placed = False
        for row in rows:
            row_y = sum(r[1] for r in row) / len(row)
            if abs(y - row_y) <= y_tol:
                row.append((x, y, h, token))
                placed = True
                break
        if not placed:
            rows.append([(x, y, h, token)])

    out: List[Tuple[float, float, float, str]] = []
    rows.sort(key=lambda row: sum(r[1] for r in row) / len(row))
    for row in rows:
        row.sort(key=lambda b: b[0])
        out.extend(row)
    return out


def letters_from_prediction(result, fail_token: str, line_y_ratio: float) -> List[str]:
    names = result.names
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return [fail_token]

    collected: List[Tuple[float, float, float, str]] = []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        label_raw = names.get(int(cls[i]), "") if isinstance(names, dict) else str(names[int(cls[i])])
        token = normalize_label(label_raw, fail_token)
        collected.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1), token))

    ordered = sort_boxes_reading_order(collected, line_y_ratio=line_y_ratio)
    letters = [token for _, _, _, token in ordered if token]
    if not letters:
        return [fail_token]
    return letters


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
                    letters = letters_from_prediction(preds[0], args.fail_token, args.line_y_ratio)
                    out_lines.append(" ".join(letters))
                except Exception as ex:
                    failed_pages += 1
                    out_lines.append(args.fail_token)
                    print(f"[WARN] {dataset_dir.name}/{page_path.name}: {ex}")
                processed_pages += 1

            out_file = dataset_dir / args.output_name
            out_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            processed_datasets += 1
            print(f"[OK] {dataset_dir.name}: wrote {out_file.name} ({len(out_lines)} lines)")

    print("=== Summary ===")
    print(f"Datasets processed: {processed_datasets}")
    print(f"Pages processed: {processed_pages}")
    print(f"Pages failed (fallback '{args.fail_token}'): {failed_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

