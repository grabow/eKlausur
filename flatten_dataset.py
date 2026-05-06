#!/usr/bin/env python3
"""
Flatten a course-structured dataset into numbered folders to remove matrikel numbers from directory names.

Input layout (current):
  data/dataset/<subset>/<matrikel>/*

Output layout (new):
  data/dataset/1/*
  data/dataset/2/*
  ...

Notes:
- Copies files (does not delete originals).
- Only copies per-student numeric folders; ignores other files like collect_report.csv.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flatten dataset into data/dataset/1..N folders (copy only).")
    p.add_argument("--dataset-root", default="data/dataset", help="Dataset root containing subset folders.")
    p.add_argument("--start-index", type=int, default=1, help="Start numbering at this index.")
    p.add_argument("--dry-run", action="store_true", help="Plan only, do not write/copy.")
    return p.parse_args()


def iter_student_dirs(dataset_root: Path) -> List[Path]:
    out: List[Path] = []
    if not dataset_root.exists():
        return out
    for subset in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        # skip already-flattened numeric dirs at root
        if subset.name.isdigit():
            continue
        for student in sorted([p for p in subset.iterdir() if p.is_dir() and p.name.isdigit()]):
            out.append(student)
    return out


def copy_tree(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(item, target)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    students = iter_student_dirs(dataset_root)
    if not students:
        print(f"[ERROR] No student folders found under {dataset_root}")
        return 2

    # Pre-flight: ensure target indices don't already exist
    idx = args.start_index
    for _ in students:
        target = dataset_root / str(idx)
        if target.exists():
            print(f"[ERROR] Target already exists: {target}")
            return 2
        idx += 1

    copied = 0
    idx = args.start_index
    for student in students:
        target = dataset_root / str(idx)
        copy_tree(student, target, args.dry_run)
        copied += 1
        idx += 1

    print("=== Flatten Summary ===")
    print(f"source student dirs: {len(students)}")
    print(f"created/planned numbered dirs: {copied}")
    print(f"output root: {dataset_root}")
    if args.dry_run:
        print("Mode: dry-run (no copies written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

