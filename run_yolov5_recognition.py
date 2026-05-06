#!/usr/bin/env python3
"""
Generate recognition_yolov5.txt for each dataset folder.

For each folder data/dataset/<id>/ that contains studSolution.txt:
- process page_*.jpg in numeric order
- run YOLOv5 recognition via existing py_yolo scripts
- write one line per image to recognition_yolov5.txt
  (letters separated by single spaces, fallback '?')
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


PAGE_RE = re.compile(r"^page_(\d+)\.jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate recognition_yolov5.txt files per dataset.")
    p.add_argument(
        "--dataset-root",
        default="data/dataset",
        help="Root folder containing numeric dataset folders.",
    )
    p.add_argument(
        "--yolo-root",
        default="/Users/wiggel/Python/py_yolo/yolov5",
        help="Folder containing get_results_yolo_all.py and dependencies.",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Optional YOLOv5 checkpoint path override (best.pt).",
    )
    p.add_argument(
        "--output-name",
        default="recognition_yolov5.txt",
        help="Output filename per dataset folder.",
    )
    p.add_argument(
        "--fail-token",
        default="?",
        help="Token to use when no letters are recognized or on page-level error.",
    )
    p.add_argument(
        "--cleanup-box-png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete temporary YOLO box crops (*_box_*.png) after processing (default: true).",
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


def normalize_letter(raw: str, fail_token: str) -> str:
    if not raw:
        return fail_token
    if raw == "?":
        return "?"
    if len(raw) == 1 and raw.isalpha():
        return raw.upper()
    return fail_token


def letters_from_result_json(result_json: str, fail_token: str) -> List[str]:
    data = json.loads(result_json)
    letters: List[str] = []
    for item in data:
        letter_obj = item.get("letter", {})
        if not isinstance(letter_obj, dict):
            continue
        letter_raw = letter_obj.get("letter")
        if not isinstance(letter_raw, str):
            continue
        letters.append(normalize_letter(letter_raw, fail_token))
    if not letters:
        return [fail_token]
    return letters


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
    out = Image.fromarray(inv)
    out.save(dst_path, quality=100, subsampling=0)


def cleanup_box_pngs(folder: Path) -> int:
    removed = 0
    for p in folder.glob("*_box_*.png"):
        try:
            p.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    yolo_root = Path(args.yolo_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not yolo_root.exists():
        raise FileNotFoundError(f"YOLO root not found: {yolo_root}")

    # YOLOv5 legacy checkpoints may fail on torch>=2.6 where weights_only=True is default.
    # We explicitly fall back to weights_only=False for trusted local checkpoints.
    import torch

    orig_torch_load = torch.load

    def patched_torch_load(*load_args, **load_kwargs):
        if "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = False
        return orig_torch_load(*load_args, **load_kwargs)

    torch.load = patched_torch_load

    sys.path.insert(0, str(yolo_root))
    from get_results_yolo_all import get_result_list  # type: ignore
    import find_boxes  # type: ignore

    if args.model_path:
        model_path = str(Path(args.model_path).expanduser().resolve())
        find_boxes.model_path = model_path
        find_boxes.model = None
        print(f"[INFO] Overriding YOLO model path: {model_path}")
    else:
        print(f"[INFO] Using YOLO model path from find_boxes.py: {find_boxes.model_path}")

    processed_datasets = 0
    processed_pages = 0
    failed_pages = 0

    removed_box_pngs = 0

    with tempfile.TemporaryDirectory(prefix="yolo_preproc_") as tmp_dir_str:
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
                    res_json = get_result_list(str(prep_path))
                    letters = letters_from_result_json(res_json, args.fail_token)
                    out_lines.append(" ".join(letters))
                except Exception as ex:
                    failed_pages += 1
                    out_lines.append(args.fail_token)
                    print(f"[WARN] {dataset_dir.name}/{page_path.name}: {ex}")
                processed_pages += 1

            out_file = dataset_dir / args.output_name
            out_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            if args.cleanup_box_png:
                removed_box_pngs += cleanup_box_pngs(dataset_dir)
            processed_datasets += 1
            print(f"[OK] {dataset_dir.name}: wrote {out_file.name} ({len(out_lines)} lines)")

    print("=== Summary ===")
    print(f"Datasets processed: {processed_datasets}")
    print(f"Pages processed: {processed_pages}")
    print(f"Pages failed (fallback '{args.fail_token}'): {failed_pages}")
    if args.cleanup_box_png:
        print(f"Temporary box PNGs removed: {removed_box_pngs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
