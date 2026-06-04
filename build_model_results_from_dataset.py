#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

DATASET_IDS = list(range(1, 62))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy model outputs from data/dataset into Results/<model>/ and write _summary.txt")
    p.add_argument("--dataset-root", default="data/dataset")
    p.add_argument("--results-root", default="Results")
    p.add_argument("--model-name", required=True)
    p.add_argument("--source-file", required=True)
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--provider-model", default="")
    p.add_argument("--copy", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def tokenize(line: str) -> List[str]:
    return [t.strip().upper() for t in line.split() if t.strip()]


def parse_result_list(line: str) -> List[str]:
    m = re.search(r"\[(.*)\]", line)
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    return [t.strip().strip("'\"").upper() for t in inner.split(",")]


def main() -> int:
    args = parse_args()
    base = Path(__file__).resolve().parent
    dataset_root = (base / args.dataset_root).resolve()
    results_root = (base / args.results_root).resolve()
    model_dir = results_root / args.model_name

    copied = 0
    if args.copy:
        for ds in DATASET_IDS:
            src = dataset_root / str(ds) / args.source_file
            if not src.exists():
                continue
            dst = model_dir / str(ds) / args.source_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1

    pages_processed = 0
    letters_total = 0
    letters_correct = 0
    extra_pred_tokens = 0

    tp = fn = fp = tn = 0

    for ds in DATASET_IDS:
        ds_dir = dataset_root / str(ds)
        gt_path = ds_dir / "studSolution.txt"
        result_path = ds_dir / "result.txt"
        pred_path = model_dir / str(ds) / args.source_file
        if not (gt_path.exists() and result_path.exists() and pred_path.exists()):
            continue

        gt_lines = gt_path.read_text(encoding="utf-8").splitlines()
        pred_lines = pred_path.read_text(encoding="utf-8").splitlines()

        pages_processed += len(gt_lines)
        for idx, gt_line in enumerate(gt_lines):
            gt = tokenize(gt_line)
            pred = tokenize(pred_lines[idx]) if idx < len(pred_lines) else []
            letters_total += len(gt)
            for j, g in enumerate(gt):
                p = pred[j] if j < len(pred) else "?"
                if p == g:
                    letters_correct += 1
            if len(pred) > len(gt):
                extra_pred_tokens += len(pred) - len(gt)

        gt_line_tokens = [tokenize(ln) for ln in gt_lines]
        pred_line_tokens = [tokenize(ln) for ln in pred_lines]

        rlines = result_path.read_text(encoding="utf-8").splitlines()
        ref_flat: List[str] = []
        gt_pos_flat: List[bool] = []
        i = 0
        while i < len(rlines):
            line = rlines[i].strip()
            if line.startswith("Referenz-Lösung"):
                j = i + 1
                while j < len(rlines) and "Lösung:" not in rlines[j]:
                    j += 1
                ref = parse_result_list(rlines[j]) if j < len(rlines) else []

                k = j + 1
                while k < len(rlines) and not rlines[k].strip().startswith("Student-Lösung"):
                    k += 1
                l = k + 1
                while l < len(rlines) and "Lösung:" not in rlines[l]:
                    l += 1
                stud = parse_result_list(rlines[l]) if l < len(rlines) else []

                m = l + 1
                while m < len(rlines) and "Punkte:" not in rlines[m]:
                    m += 1
                pts = parse_result_list(rlines[m]) if m < len(rlines) else []

                n = min(len(ref), len(stud))
                if pts and len(pts) >= n:
                    gt = [(p not in ("0", "0.0")) for p in pts[:n]]
                else:
                    gt = [stud[x] == ref[x] for x in range(n)]

                ref_flat.extend(ref[:n])
                gt_pos_flat.extend(gt)
                i = m
            i += 1

        idx = 0
        for line_idx, gt_tokens in enumerate(gt_line_tokens):
            line_len = len(gt_tokens)
            ref_line = ref_flat[idx:idx + line_len]
            gt_pos_line = gt_pos_flat[idx:idx + line_len]
            idx += line_len

            pred_line = pred_line_tokens[line_idx] if line_idx < len(pred_line_tokens) else []
            for token_idx in range(line_len):
                if token_idx >= len(ref_line) or token_idx >= len(gt_pos_line):
                    continue
                ref = ref_line[token_idx]
                gt_pos = gt_pos_line[token_idx]
                pred = pred_line[token_idx] if token_idx < len(pred_line) else "?"
                pred_pos = pred == ref

                if gt_pos and pred_pos:
                    tp += 1
                elif gt_pos and not pred_pos:
                    fn += 1
                elif (not gt_pos) and pred_pos:
                    fp += 1
                else:
                    tn += 1

    letters_wrong = letters_total - letters_correct
    accuracy = (letters_correct / letters_total) if letters_total else 0.0

    pos = tp + fn
    neg = tn + fp
    recall = (tp / pos) if pos else 0.0
    fnr = (fn / pos) if pos else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    specificity = (tn / neg) if neg else 0.0

    datasets_copied = len([ds for ds in DATASET_IDS if (model_dir / str(ds) / args.source_file).exists()])

    summary_path = model_dir / "_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"model: {args.model_name}",
        f"source_file: {args.source_file}",
        f"datasets_expected: 61",
        f"datasets_copied: {datasets_copied}",
        f"pages_processed: {pages_processed}",
        f"letters_total: {letters_total}",
        f"letters_correct: {letters_correct}",
        f"letters_wrong: {letters_wrong}",
        f"accuracy: {accuracy:.6f}",
        f"accuracy_percent: {accuracy*100.0:.2f}",
        f"extra_pred_tokens: {extra_pred_tokens}",
        f"tp: {tp}",
        f"fn: {fn}",
        f"fp: {fp}",
        f"tn: {tn}",
        f"recall: {recall:.6f}",
        f"fnr: {fnr:.6f}",
        f"precision: {precision:.6f}",
        f"specificity: {specificity:.6f}",
        f"provider: {args.provider}",
        f"provider_model: {args.provider_model}",
        f"created_at: {datetime.now(timezone.utc).isoformat()}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote summary: {summary_path}")
    print(f"[OK] datasets_copied: {datasets_copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
