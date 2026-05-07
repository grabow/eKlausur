#!/usr/bin/env python3
"""
Normalize solutions and renumber page images per student folder.

Requirements (per student folder):
- Transform studSolution.txt into only letter lines:
  - remove empty lines
  - remove lines starting with '#'
  - keep only A-Z single-letter tokens (uppercased) joined by single spaces
- Write the same content to studSolution.yaml
  (line 1 corresponds to the first image after renumbering)
- Renumber kept images so they become contiguous:
  - Order is defined by structure.yaml: 'seite: X' => original image 'page_(X-1).jpg'
  - After renumbering: the first referenced image becomes page_0.jpg, then page_1.jpg, ...
- Update structure.yaml so its seite numbers become 1..N (preserving order),
  keeping the invariant: seite X <-> page_(X-1).jpg.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


SEITE_LINE_RE = re.compile(r"^(\s*-\s*seite\s*:\s*)(\d+)(\s*)$", re.IGNORECASE)
LETTER_OR_UNKNOWN_TOKEN_RE = re.compile(r"^(?:[A-Za-z]|\?)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize studSolution and renumber page images.")
    p.add_argument(
        "--dataset-roots",
        nargs="+",
        default=["data/dataset"],
        help="One or more dataset roots (each contains course folders).",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only, do not modify files.")
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


def parse_structure_order(structure_path: Path) -> Tuple[List[int], List[str]]:
    """
    Returns:
    - page_indices: in-order list of required page indices (seite-1)
    - lines: raw lines of the structure file (with line endings preserved)
    """
    text = structure_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    page_indices: List[int] = []
    for ln in lines:
        m = SEITE_LINE_RE.match(ln.rstrip("\n\r"))
        if not m:
            continue
        seite = int(m.group(2))
        if seite > 0:
            page_indices.append(seite - 1)
    return page_indices, lines


def rewrite_structure_seiten_sequential(lines: Sequence[str]) -> Tuple[str, int]:
    out_lines: List[str] = []
    next_seite = 1
    replaced = 0
    for ln in lines:
        m = SEITE_LINE_RE.match(ln.rstrip("\n\r"))
        if not m:
            out_lines.append(ln)
            continue
        prefix, _old, suffix = m.group(1), m.group(2), m.group(3)
        newline = "\n" if ln.endswith("\n") else ""
        out_lines.append(f"{prefix}{next_seite}{suffix}{newline}")
        next_seite += 1
        replaced += 1
    return "".join(out_lines), replaced


def normalize_solution_lines(stud_solution_path: Path) -> List[str]:
    text = stud_solution_path.read_text(encoding="utf-8", errors="replace")
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        tokens = [t.strip() for t in line.split() if t.strip()]
        normalized: List[str] = []
        for t in tokens:
            if not LETTER_OR_UNKNOWN_TOKEN_RE.match(t):
                continue
            if t == "?":
                normalized.append("?")
            else:
                normalized.append(t.upper())
        if not normalized:
            continue
        out.append(" ".join(normalized))
    return out


def find_page_file(student_dir: Path, page_index: int) -> Path | None:
    # Prefer non-underscore, but accept underscore variant.
    for stem in (f"page_{page_index}", f"_page_{page_index}"):
        candidate = student_dir / f"{stem}.jpg"
        if candidate.exists():
            return candidate
    return None


def safe_renumber_pages(student_dir: Path, ordered_page_indices: Sequence[int], dry_run: bool) -> int:
    """
    Renames referenced pages to contiguous page_0.jpg..page_{n-1}.jpg
    using a two-phase tmp rename to avoid collisions.
    Returns number of pages renamed.
    """
    mapping: List[Tuple[Path, Path, Path]] = []
    for new_idx, old_idx in enumerate(ordered_page_indices):
        src = find_page_file(student_dir, old_idx)
        if src is None:
            raise FileNotFoundError(f"Missing required image for page index {old_idx} in {student_dir}")
        tmp = student_dir / f".__tmp_page_{new_idx}__{src.name}"
        dst = student_dir / f"page_{new_idx}.jpg"
        mapping.append((src, tmp, dst))

    if dry_run:
        return len(mapping)

    # Phase 1: src -> tmp
    for src, tmp, _dst in mapping:
        if tmp.exists():
            tmp.unlink()
        src.rename(tmp)

    # Phase 2: tmp -> dst
    for _src, tmp, dst in mapping:
        if dst.exists():
            dst.unlink()
        tmp.rename(dst)

    return len(mapping)


def write_text(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_roots = [Path(p).expanduser().resolve() for p in args.dataset_roots]

    processed = 0
    renamed_pages = 0
    updated_solutions = 0
    updated_structures = 0
    skipped = 0

    for student_dir in iter_student_dirs(dataset_roots):
        structure = student_dir / "structure.yaml"
        stud_solution = student_dir / "studSolution.txt"
        if not structure.exists() or not stud_solution.exists():
            skipped += 1
            continue

        ordered_page_indices, structure_lines = parse_structure_order(structure)
        solution_lines = normalize_solution_lines(stud_solution)

        if len(solution_lines) != len(ordered_page_indices):
            raise RuntimeError(
                f"Line/page mismatch in {student_dir}: "
                f"{len(solution_lines)} solution lines vs {len(ordered_page_indices)} pages in structure.yaml"
            )

        # 1) rewrite solutions (txt + yaml)
        sol_text = "\n".join(solution_lines) + "\n"
        write_text(stud_solution, sol_text, args.dry_run)
        write_text(student_dir / "studSolution.yaml", sol_text, args.dry_run)
        updated_solutions += 1

        # 2) renumber images
        renamed_pages += safe_renumber_pages(student_dir, ordered_page_indices, args.dry_run)

        # 3) rewrite structure.yaml seite numbers sequentially
        new_structure, replaced = rewrite_structure_seiten_sequential(structure_lines)
        write_text(structure, new_structure, args.dry_run)
        if replaced:
            updated_structures += 1

        processed += 1

    print("=== Normalize + Renumber Summary ===")
    print(f"student dirs processed: {processed}")
    print(f"student dirs skipped (missing files): {skipped}")
    print(f"solution files updated/planned: {updated_solutions}")
    print(f"page images renamed/planned: {renamed_pages}")
    print(f"structures updated/planned: {updated_structures}")
    if args.dry_run:
        print("Mode: dry-run (no changes written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
