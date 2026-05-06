#!/usr/bin/env python3
"""
Anonymize collected exam dataset folders in-place.

Operations (per student folder):
1) Delete klausur.pdf
2) In result.txt remove the first two header lines:
   - "matrikelnummer: ..."
   - "points: ..."
3) In studSolution.txt remove the first two header lines:
   - "<digits> #Matrikelnummer"
   - "<letter> #Bogen"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple


RESULT_HDR_1 = re.compile(r"^\s*matrikelnummer\s*:\s*\S+.*$", re.IGNORECASE)
RESULT_HDR_2 = re.compile(r"^\s*points\s*:\s*\S+.*$", re.IGNORECASE)
STUD_HDR_1 = re.compile(r"^\s*\d+\s*#\s*matrikelnummer\s*$", re.IGNORECASE)
STUD_HDR_2 = re.compile(r"^\s*[a-z]\s*#\s*bogen\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anonymize dataset in-place.")
    p.add_argument(
        "--dataset-roots",
        nargs="+",
        default=["data/dataset"],
        help="One or more dataset roots (each contains course folders).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions but do not modify files.")
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


def delete_file(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        return True
    path.unlink()
    return True


def strip_two_header_lines(path: Path, hdr1: re.Pattern[str], hdr2: re.Pattern[str], dry_run: bool) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if len(lines) < 2:
        return False, "too_short"

    if not hdr1.match(lines[0].rstrip("\n\r")) or not hdr2.match(lines[1].rstrip("\n\r")):
        return False, "header_mismatch"

    new_text = "".join(lines[2:])
    if dry_run:
        return True, "planned"
    path.write_text(new_text, encoding="utf-8")
    return True, "updated"


def main() -> int:
    args = parse_args()
    dataset_roots = [Path(p).expanduser().resolve() for p in args.dataset_roots]

    pdf_deleted = 0
    result_updated = 0
    result_skipped = 0
    stud_updated = 0
    stud_skipped = 0

    for student_dir in iter_student_dirs(dataset_roots):
        pdf = student_dir / "klausur.pdf"
        if delete_file(pdf, args.dry_run):
            pdf_deleted += 1

        ok, status = strip_two_header_lines(student_dir / "result.txt", RESULT_HDR_1, RESULT_HDR_2, args.dry_run)
        if ok:
            result_updated += 1
        else:
            result_skipped += 1

        ok, status = strip_two_header_lines(student_dir / "studSolution.txt", STUD_HDR_1, STUD_HDR_2, args.dry_run)
        if ok:
            stud_updated += 1
        else:
            stud_skipped += 1

    print("=== Anonymization Summary ===")
    print(f"klausur.pdf deleted/planned: {pdf_deleted}")
    print(f"result.txt updated/planned: {result_updated} (skipped: {result_skipped})")
    print(f"studSolution.txt updated/planned: {stud_updated} (skipped: {stud_skipped})")
    if args.dry_run:
        print("Mode: dry-run (no changes written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

