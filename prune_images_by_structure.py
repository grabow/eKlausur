#!/usr/bin/env python3
"""
Delete unused page images from dataset student folders based on structure.yaml.

Rule: In structure.yaml, 'seite: X' corresponds to image file 'page_(X-1).*'.
All other page images (page_*.jpg/jpeg/png and _page_*.jpg/jpeg/png) are removed.

This script operates in-place.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Set, Tuple


SEITE_RE = re.compile(r"^\s*-\s*seite\s*:\s*(\d+)\s*$", re.IGNORECASE)
PAGE_RE = re.compile(r"^_?page_(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune unused page images using structure.yaml.")
    p.add_argument(
        "--dataset-roots",
        nargs="+",
        default=["data/dataset"],
        help="One or more dataset roots (each contains course folders).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print summary but do not delete files.")
    return p.parse_args()


def iter_student_dirs(dataset_roots: Iterable[Path]) -> Iterable[Path]:
    for root in dataset_roots:
        if not root.exists():
            continue
        if root.is_dir() and root.name.isdigit():
            yield root
            continue
        for course_dir in root.iterdir():
            if not course_dir.is_dir():
                continue
            for student_dir in course_dir.iterdir():
                if student_dir.is_dir() and student_dir.name.isdigit():
                    yield student_dir


def required_page_indices_from_structure(structure_path: Path) -> Set[int]:
    text = structure_path.read_text(encoding="utf-8", errors="replace")
    required: Set[int] = set()
    for line in text.splitlines():
        m = SEITE_RE.match(line)
        if not m:
            continue
        seite = int(m.group(1))
        if seite <= 0:
            continue
        required.add(seite - 1)
    return required


def is_page_image(path: Path) -> Tuple[bool, int]:
    if not path.is_file():
        return False, -1
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return False, -1
    m = PAGE_RE.match(path.stem)
    if not m:
        return False, -1
    return True, int(m.group(1))


def main() -> int:
    args = parse_args()
    dataset_roots = [Path(p).expanduser().resolve() for p in args.dataset_roots]

    total_imgs = 0
    kept_imgs = 0
    deleted_imgs = 0
    missing_structure = 0

    for student_dir in iter_student_dirs(dataset_roots):
        structure = student_dir / "structure.yaml"
        if not structure.exists():
            missing_structure += 1
            continue

        required = required_page_indices_from_structure(structure)
        for img in student_dir.iterdir():
            is_page, idx = is_page_image(img)
            if not is_page:
                continue
            total_imgs += 1
            if idx in required:
                kept_imgs += 1
                continue
            deleted_imgs += 1
            if not args.dry_run:
                img.unlink()

    print("=== Image Prune Summary ===")
    print(f"page images found: {total_imgs}")
    print(f"kept (in structure.yaml): {kept_imgs}")
    print(f"deleted/planned: {deleted_imgs}")
    if missing_structure:
        print(f"student dirs missing structure.yaml (skipped): {missing_structure}")
    if args.dry_run:
        print("Mode: dry-run (no deletions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

