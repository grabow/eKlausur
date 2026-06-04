#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DATASET_IDS = list(range(1, 62))


@dataclass
class ModelSpec:
    name: str
    source_file: str
    model_dir: Path


def parse_summary(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def parse_result_list(line: str) -> List[str]:
    m = re.search(r"\[(.*)\]", line)
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    return [t.strip().strip("'\"").upper() for t in inner.split(",")]


def tokenize(line: str) -> List[str]:
    return [t.strip().upper() for t in line.split() if t.strip()]


def load_model_specs(results_root: Path) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for summary_path in sorted(results_root.glob("*/_summary.txt")):
        data = parse_summary(summary_path)
        name = data.get("model")
        source_file = data.get("source_file")
        if not name or not source_file:
            continue
        specs.append(ModelSpec(name=name, source_file=source_file, model_dir=summary_path.parent))
    return specs


def evaluate_model(dataset_root: Path, spec: ModelSpec) -> Dict[str, float]:
    tp = fn = fp = tn = 0
    covered_datasets = 0

    for ds in DATASET_IDS:
        ds_dir = dataset_root / str(ds)
        gt_path = ds_dir / "studSolution.txt"
        result_path = ds_dir / "result.txt"
        pred_path = spec.model_dir / str(ds) / spec.source_file
        if not (gt_path.exists() and result_path.exists() and pred_path.exists()):
            continue

        covered_datasets += 1
        gt_line_tokens = [tokenize(ln) for ln in gt_path.read_text(encoding="utf-8").splitlines()]
        pred_line_tokens = [tokenize(ln) for ln in pred_path.read_text(encoding="utf-8").splitlines()]

        # Build flat reference tokens and GT positivity labels from result.txt
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

        # Slice by studSolution line lengths to avoid shift due to extra predicted tokens.
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

    pos = tp + fn
    neg = tn + fp
    recall = (tp / pos) if pos else 0.0
    fnr = (fn / pos) if pos else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    specificity = (tn / neg) if neg else 0.0
    accuracy = ((tp + tn) / (pos + neg)) if (pos + neg) else 0.0
    return {
        "datasets_covered": covered_datasets,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "fnr": fnr,
        "precision": precision,
        "specificity": specificity,
        "classification_accuracy": accuracy,
    }


def main() -> int:
    base = Path(__file__).resolve().parent
    dataset_root = base / "data" / "dataset"
    results_root = base / "Results"
    out_csv = results_root / "classification_metrics_all_results.csv"
    out_md = results_root / "classification_metrics_all_results.md"

    specs = load_model_specs(results_root)
    if not specs:
        raise RuntimeError(f"No summaries found under {results_root}")

    rows: List[List[object]] = []
    for spec in specs:
        metrics = evaluate_model(dataset_root, spec)
        rows.append(
            [
                spec.name,
                spec.source_file,
                int(metrics["datasets_covered"]),
                int(metrics["tp"]),
                int(metrics["fn"]),
                int(metrics["fp"]),
                int(metrics["tn"]),
                f"{metrics['recall']:.6f}",
                f"{metrics['fnr']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['specificity']:.6f}",
                f"{metrics['classification_accuracy']:.6f}",
            ]
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "source_file",
                "datasets_covered",
                "tp",
                "fn",
                "fp",
                "tn",
                "recall",
                "fnr",
                "precision",
                "specificity",
                "classification_accuracy",
            ]
        )
        w.writerows(rows)

    rows_sorted = sorted(rows, key=lambda r: float(r[8]))  # by FNR asc
    lines = [
        "# Classification Metrics For All Results",
        "",
        "| Model | TP | FN | FP | TN | Recall | FNR | Precision | Specificity | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(f"| {r[0]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | {r[9]} | {r[10]} | {r[11]} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
