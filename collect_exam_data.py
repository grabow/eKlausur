#!/usr/bin/env python3
"""
Collect consented exam folders from multiple source roots.

Features:
- Reads consent tables from CSV/TSV/XLSX (no third-party dependencies).
- Extracts matriculation numbers from configured columns or by regex fallback.
- Copies matching student folders (numeric folder names) into one target directory.
- Handles duplicates across source roots with configurable strategy.
- Writes a machine-readable report CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import xml.etree.ElementTree as ET

MATR_REGEX = re.compile(r"\b\d{5,}\b")
EMAIL_REGEX = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass
class Candidate:
    matrikel: str
    source_root: Path
    folder: Path
    mtime: float


@dataclass
class CopyResult:
    matrikel: str
    status: str
    source: str
    target: str
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect consented exam folders from multiple source roots.")
    p.add_argument("--consent-files", nargs="+", required=True, help="Consent tables (CSV/TSV/XLSX).")
    p.add_argument("--source-roots", nargs="+", required=True, help="Directories containing matrikel folders.")
    p.add_argument("--out-dir", required=True, help="Target directory for collected folders.")
    p.add_argument(
        "--matrikel-columns",
        default="matrikel,matrikelnummer,idnumber,id-nummer,student_id,matrnr",
        help="Comma-separated candidate column names for matrikel numbers.",
    )
    p.add_argument(
        "--email-columns",
        default="email,e-mail,mail,nutzername,username",
        help="Comma-separated candidate column names for email addresses (used when matrikel is missing).",
    )
    p.add_argument("--min-digits", type=int, default=5, help="Minimum digits for a valid matrikel number.")
    p.add_argument(
        "--required-files",
        default="klausur.pdf,studSolution.txt",
        help="Comma-separated file names that must exist in source folder.",
    )
    p.add_argument(
        "--on-duplicate",
        choices=["first", "newest", "error"],
        default="first",
        help="How to resolve a matrikel found in multiple source roots.",
    )
    p.add_argument(
        "--copy-mode",
        choices=["full", "minimal"],
        default="full",
        help="Copy full folder or only selected files.",
    )
    p.add_argument(
        "--include-patterns",
        default="klausur.pdf,studSolution.txt,correctSolution.yaml,result.txt,page_*.jpg,page_*.jpeg,page_*.png,_page_*.jpg,_page_*.jpeg,_page_*.png",
        help="Comma-separated glob patterns used when --copy-mode minimal.",
    )
    p.add_argument("--dry-run", action="store_true", help="Scan and plan only, no copy.")
    p.add_argument(
        "--report-csv",
        default="collect_report.csv",
        help="Report CSV filename (written into out-dir unless absolute path).",
    )
    return p.parse_args()


def normalize_matrikel(value: str, min_digits: int) -> Optional[str]:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    m = re.search(rf"\d{{{min_digits},}}", txt)
    if not m:
        return None
    return m.group(0)


def split_csv_line(line: str, delimiter: str) -> List[str]:
    reader = csv.reader([line], delimiter=delimiter)
    return next(reader)


def detect_delimiter(sample: str) -> str:
    counts = {
        ",": sample.count(","),
        ";": sample.count(";"),
        "\t": sample.count("\t"),
    }
    # pick the delimiter with the highest count; fall back to comma
    return max(counts, key=counts.get) if any(counts.values()) else ","


def sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        return dialect.delimiter
    except Exception:
        return detect_delimiter(sample)


def read_table_rows(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return read_delimited_rows(path)
    if suffix == ".xlsx":
        return read_xlsx_rows(path)
    raise ValueError(f"Unsupported consent table format: {path}")


def read_delimited_rows(path: Path) -> List[Dict[str, str]]:
    # Use csv module to properly handle quoting and embedded newlines.
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(4096)
        if not sample.strip():
            return []
        delimiter = sniff_delimiter(sample)
        f.seek(0)

        reader = csv.reader(f, delimiter=delimiter)

        header: Optional[List[str]] = None
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            header = [c.strip() for c in row]
            break
        if not header:
            return []

        rows: List[Dict[str, str]] = []
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            out: Dict[str, str] = {}
            for i, key in enumerate(header):
                if not key:
                    key = f"col_{i+1}"
                out[key] = row[i].strip() if i < len(row) and row[i] is not None else ""
            rows.append(out)
        return rows


def read_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    with zipfile.ZipFile(path, "r") as zf:
        shared_strings = parse_shared_strings(zf)
        sheet_paths = parse_sheet_paths(zf)
        if not sheet_paths:
            return []

        merged: List[Dict[str, str]] = []
        for sheet_path in sheet_paths:
            rows = parse_sheet_rows(zf, sheet_path, shared_strings)
            merged.extend(rows)
        return merged


def parse_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    data = zf.read("xl/sharedStrings.xml")
    root = ET.fromstring(data)
    out: List[str] = []
    for si in root.findall(f"{{{NS_MAIN}}}si"):
        parts = []
        t = si.find(f"{{{NS_MAIN}}}t")
        if t is not None and t.text is not None:
            parts.append(t.text)
        for r in si.findall(f"{{{NS_MAIN}}}r"):
            rt = r.find(f"{{{NS_MAIN}}}t")
            if rt is not None and rt.text is not None:
                parts.append(rt.text)
        out.append("".join(parts))
    return out


def parse_sheet_paths(zf: zipfile.ZipFile) -> List[str]:
    wb_name = "xl/workbook.xml"
    rels_name = "xl/_rels/workbook.xml.rels"
    if wb_name not in zf.namelist() or rels_name not in zf.namelist():
        return []

    wb_root = ET.fromstring(zf.read(wb_name))
    rel_root = ET.fromstring(zf.read(rels_name))

    rel_by_id: Dict[str, str] = {}
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    for rel in rel_root.findall(f"{{{rel_ns}}}Relationship"):
        rid = rel.attrib.get("Id", "")
        target = rel.attrib.get("Target", "")
        if rid and target:
            rel_by_id[rid] = target

    out: List[str] = []
    sheets_parent = wb_root.find(f"{{{NS_MAIN}}}sheets")
    if sheets_parent is None:
        return out

    for sheet in sheets_parent.findall(f"{{{NS_MAIN}}}sheet"):
        rid = sheet.attrib.get(f"{{{NS_REL}}}id", "")
        target = rel_by_id.get(rid, "")
        if not target:
            continue
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        out.append(target)
    return out


def parse_sheet_rows(zf: zipfile.ZipFile, sheet_path: str, shared_strings: Sequence[str]) -> List[Dict[str, str]]:
    if sheet_path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(sheet_path))
    sheet_data = root.find(f"{{{NS_MAIN}}}sheetData")
    if sheet_data is None:
        return []

    raw_rows: List[Dict[int, str]] = []
    for row in sheet_data.findall(f"{{{NS_MAIN}}}row"):
        cells: Dict[int, str] = {}
        for c in row.findall(f"{{{NS_MAIN}}}c"):
            ref = c.attrib.get("r", "")
            idx = excel_col_to_index(ref)
            cells[idx] = parse_xlsx_cell_value(c, shared_strings)
        if cells:
            raw_rows.append(cells)

    if not raw_rows:
        return []

    header_cells = raw_rows[0]
    max_col = max(header_cells.keys())
    headers = [header_cells.get(i, "").strip() for i in range(max_col + 1)]

    rows: List[Dict[str, str]] = []
    for cells in raw_rows[1:]:
        row: Dict[str, str] = {}
        max_c = max(max_col, max(cells.keys(), default=0))
        for i in range(max_c + 1):
            key = headers[i] if i < len(headers) and headers[i] else f"col_{i+1}"
            row[key] = cells.get(i, "").strip()
        rows.append(row)
    return rows


def parse_xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    v = cell.find(f"{{{NS_MAIN}}}v")
    is_node = cell.find(f"{{{NS_MAIN}}}is")

    if cell_type == "inlineStr" and is_node is not None:
        t = is_node.find(f"{{{NS_MAIN}}}t")
        return "" if t is None or t.text is None else t.text

    if v is None or v.text is None:
        return ""
    txt = v.text

    if cell_type == "s":
        try:
            idx = int(txt)
            return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
        except Exception:
            return ""

    return txt


def excel_col_to_index(ref: str) -> int:
    if not ref:
        return 0
    letters = ""
    for ch in ref:
        if ch.isalpha():
            letters += ch
        else:
            break
    if not letters:
        return 0
    out = 0
    for ch in letters.upper():
        out = out * 26 + (ord(ch) - ord("A") + 1)
    return max(0, out - 1)


def extract_matrikels_from_rows(rows: Sequence[Dict[str, str]], column_candidates: Sequence[str], min_digits: int) -> Set[str]:
    want = {c.strip().lower() for c in column_candidates if c.strip()}
    out: Set[str] = set()

    for row in rows:
        if not row:
            continue

        if not row_has_consent_yes(row):
            continue

        # 1) preferred columns
        found_from_named = False
        for key, val in row.items():
            if key.strip().lower() in want:
                m = normalize_matrikel(val, min_digits)
                if m:
                    out.add(m)
                    found_from_named = True
        if found_from_named:
            continue

        # 2) fallback: scan all values
        for val in row.values():
            if not val:
                continue
            for m in re.findall(rf"\b\d{{{min_digits},}}\b", str(val)):
                out.add(m)

    return out


def row_has_consent_yes(row: Dict[str, str]) -> bool:
    """
    If the table contains a consent column, only treat rows with an affirmative value as consented.
    Otherwise, return True (no filtering).
    """
    consent_keys = []
    for k in row.keys():
        kl = k.strip().lower()
        if any(tok in kl for tok in ("stimme", "einwill", "zustimm", "consent", "agree")):
            consent_keys.append(k)

    if not consent_keys:
        return True

    yes_values = {"ja", "yes", "true", "1", "y"}
    for k in consent_keys:
        v = (row.get(k) or "").strip().strip('"').strip().lower()
        if v in yes_values:
            return True
    return False


def extract_emails_from_rows(rows: Sequence[Dict[str, str]], column_candidates: Sequence[str]) -> Set[str]:
    want = {c.strip().lower() for c in column_candidates if c.strip()}
    out: Set[str] = set()

    for row in rows:
        if not row:
            continue

        if not row_has_consent_yes(row):
            continue

        found_from_named = False
        for key, val in row.items():
            if key.strip().lower() in want and val:
                for e in EMAIL_REGEX.findall(str(val)):
                    out.add(e.lower())
                    found_from_named = True
        if found_from_named:
            continue

        for val in row.values():
            if not val:
                continue
            for e in EMAIL_REGEX.findall(str(val)):
                out.add(e.lower())
    return out


def parse_studlist_yaml(path: Path) -> Dict[str, str]:
    """
    Minimal YAML parser for the known studList*.yaml format:
    each student block contains 'email:' and 'matrikelnummer:'.
    Returns email->matrikel mapping.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    mapping: Dict[str, str] = {}

    current_email: Optional[str] = None
    current_matr: Optional[str] = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        if line.startswith("- "):
            # Start of new student block; commit previous if complete.
            if current_email and current_matr:
                mapping[current_email.lower()] = current_matr
            current_email = None
            current_matr = None

        if line.startswith("email:"):
            val = line.split(":", 1)[1].strip().strip('"').strip("'").strip()
            if val:
                current_email = val
        elif line.startswith("matrikelnummer:"):
            val = line.split(":", 1)[1].strip().strip('"').strip("'").strip()
            m = normalize_matrikel(val, 5)
            if m:
                current_matr = m

    if current_email and current_matr:
        mapping[current_email.lower()] = current_matr
    return mapping


def build_email_to_matrikel_map(source_roots: Sequence[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for root in source_roots:
        if not root.is_dir():
            continue
        for studlist in sorted(root.glob("studList*.yaml")):
            try:
                part = parse_studlist_yaml(studlist)
                mapping.update(part)
            except Exception:
                continue
    return mapping


def list_candidates(source_root: Path, min_digits: int, required_files: Sequence[str]) -> List[Candidate]:
    out: List[Candidate] = []
    if not source_root.is_dir():
        return out

    for child in source_root.iterdir():
        if not child.is_dir():
            continue
        matr = normalize_matrikel(child.name, min_digits)
        if not matr or child.name != matr:
            continue

        missing = [rf for rf in required_files if rf and not (child / rf).exists()]
        if missing:
            continue

        try:
            mtime = child.stat().st_mtime
        except Exception:
            mtime = 0.0
        out.append(Candidate(matrikel=matr, source_root=source_root, folder=child, mtime=mtime))
    return out


def choose_candidate(existing: Candidate, new: Candidate, strategy: str) -> Candidate:
    if strategy == "first":
        return existing
    if strategy == "newest":
        return new if new.mtime > existing.mtime else existing
    if strategy == "error":
        raise RuntimeError(
            f"Duplicate matrikel across roots: {existing.matrikel} ({existing.folder}) and ({new.folder})"
        )
    return existing


def safe_copytree(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_minimal(src: Path, dst: Path, patterns: Sequence[str], dry_run: bool) -> int:
    files: List[Path] = []
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        files.extend(src.glob(pattern))

    unique = sorted({f for f in files if f.is_file()})
    if dry_run:
        return len(unique)

    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in unique:
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
    return copied


def write_report(rows: Sequence[CopyResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrikel", "status", "source", "target", "note"])
        for r in rows:
            w.writerow([r.matrikel, r.status, r.source, r.target, r.note])


def main() -> int:
    args = parse_args()

    consent_files = [Path(p).expanduser().resolve() for p in args.consent_files]
    source_roots = [Path(p).expanduser().resolve() for p in args.source_roots]
    out_dir = Path(args.out_dir).expanduser().resolve()

    required_files = [s.strip() for s in args.required_files.split(",") if s.strip()]
    include_patterns = [s.strip() for s in args.include_patterns.split(",") if s.strip()]
    matrikel_columns = [s.strip() for s in args.matrikel_columns.split(",") if s.strip()]
    email_columns = [s.strip() for s in args.email_columns.split(",") if s.strip()]

    report_path = Path(args.report_csv)
    if not report_path.is_absolute():
        report_path = out_dir / report_path

    # 1) Load consent list
    consent_ids: Set[str] = set()
    consent_emails: Set[str] = set()
    for cf in consent_files:
        if not cf.exists():
            print(f"[WARN] consent file not found: {cf}")
            continue
        try:
            rows = read_table_rows(cf)
            ids = extract_matrikels_from_rows(rows, matrikel_columns, args.min_digits)
            emails = extract_emails_from_rows(rows, email_columns)
            print(
                f"[INFO] {cf.name}: loaded {len(rows)} rows, found {len(ids)} matrikel IDs, {len(emails)} emails"
            )
            consent_ids.update(ids)
            consent_emails.update(emails)
        except Exception as ex:
            print(f"[WARN] failed to parse consent file {cf}: {ex}")

    if not consent_ids and not consent_emails:
        print("[ERROR] No consented matrikel IDs or emails found. Aborting.")
        return 2

    if not consent_ids and consent_emails:
        email_map = build_email_to_matrikel_map(source_roots)
        for e in sorted(consent_emails):
            mid = email_map.get(e.lower())
            if mid:
                consent_ids.add(mid)
        if not consent_ids:
            print("[ERROR] Consented emails found, but no email->matrikel mapping could be resolved. Aborting.")
            return 2
        print(f"[INFO] Resolved {len(consent_ids)} matrikel IDs from consent emails via studList*.yaml")

    # 2) Scan source roots
    chosen: Dict[str, Candidate] = {}
    for root in source_roots:
        cands = list_candidates(root, args.min_digits, required_files)
        print(f"[INFO] {root}: found {len(cands)} eligible folders")
        for cand in cands:
            if cand.matrikel not in consent_ids:
                continue
            if cand.matrikel not in chosen:
                chosen[cand.matrikel] = cand
            else:
                chosen[cand.matrikel] = choose_candidate(chosen[cand.matrikel], cand, args.on_duplicate)

    # 3) Copy
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: List[CopyResult] = []
    missing = sorted(consent_ids - set(chosen.keys()))
    for mid in missing:
        results.append(CopyResult(mid, "missing", "", "", "No matching source folder found"))

    copied_count = 0
    for mid in sorted(chosen.keys()):
        cand = chosen[mid]
        target = out_dir / mid
        try:
            if args.copy_mode == "full":
                safe_copytree(cand.folder, target, args.dry_run)
                note = "copied full folder"
            else:
                num = copy_minimal(cand.folder, target, include_patterns, args.dry_run)
                note = f"copied minimal files: {num}"
            copied_count += 1
            results.append(CopyResult(mid, "copied" if not args.dry_run else "planned", str(cand.folder), str(target), note))
        except Exception as ex:
            results.append(CopyResult(mid, "error", str(cand.folder), str(target), str(ex)))

    write_report(results, report_path)

    total = len(consent_ids)
    found = len(chosen)
    print("\n=== Collection Summary ===")
    print(f"Consented IDs: {total}")
    print(f"Matched in sources: {found}")
    print(f"Copied/Planned: {copied_count}")
    print(f"Missing: {len(missing)}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
