#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


DATASET_IDS = list(range(1, 11))


@dataclass
class ModelSpec:
    name: str
    source_file: str


MODELS: List[ModelSpec] = [
    ModelSpec("LLM OpenRouter Gemini-3.1 Flash-Lite-plain", "recognition_llm_openrouter_gemini31_flashlite.txt"),
    ModelSpec("LLM OpenRouter Gemini-3.1 Flash-Lite-weak", "recognition_llm_openrouter_gemini31_flashlite_weak_hint_ref.txt"),
    ModelSpec("LLM Google Gemini-3.5 Flash-plain", "recognition_llm_gemini35_flash_google_plain.txt"),
    ModelSpec("LLM Google Gemini-3.5 Flash-weak", "recognition_llm_gemini35_flash_google_weak_hint_ref.txt"),
    ModelSpec("LLM OpenRouter OpenAI GPT-5.2-plain", "recognition_llm_openrouter_openai_gpt52_plain.txt"),
    ModelSpec("LLM OpenRouter OpenAI GPT-5.2-weak", "recognition_llm_openrouter_openai_gpt52_weak_hint_ref.txt"),
    ModelSpec("LLM OpenRouter OpenAI GPT-5.5-plain-subset_1_10", "recognition_llm_openrouter_openai_gpt55_plain.txt"),
    ModelSpec("LLM OpenRouter OpenAI GPT-5.5-weak-subset_1_10", "recognition_llm_openrouter_openai_gpt55_weak_hint_ref.txt"),
    ModelSpec("LLM OpenRouter Anthropic Claude Opus 4.7-plain-subset_1_10", "recognition_llm_openrouter_claude_opus47_plain.txt"),
    ModelSpec("LLM OpenRouter Anthropic Claude Opus 4.7-weak-subset_1_10", "recognition_llm_openrouter_claude_opus47_weak_hint_ref.txt"),
    ModelSpec("LLM OpenRouter xAI Grok-4.3-plain", "recognition_llm_grok43_plain.txt"),
    ModelSpec("LLM OpenRouter xAI Grok-4.3-weak", "recognition_llm_grok43_weak_hint_ref.txt"),
    ModelSpec("LLM OpenRouter Qwen3-VL-30B-plain", "recognition_llm_qwen3vl30b_plain.txt"),
    ModelSpec("LLM OpenRouter Qwen3-VL-30B-weak", "recognition_llm_qwen3vl30b_weak_hint_ref.txt"),
    ModelSpec("YOLOv5 Medium-plain", "recognition_yolov5_medium.txt"),
    ModelSpec("YOLO26 Medium-plain", "recognition_yolo26_medium.txt"),
]


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


def evaluate_model(dataset_root: Path, spec: ModelSpec) -> Dict[str, float]:
    datasets_covered = 0
    pages_processed = 0
    letters_total = 0
    letters_correct = 0
    extra_pred_tokens = 0

    tp = fn = fp = tn = 0
    q_tokens = 0
    q_lines = 0

    for ds in DATASET_IDS:
        ds_dir = dataset_root / str(ds)
        gt_path = ds_dir / "studSolution.txt"
        result_path = ds_dir / "result.txt"
        pred_path = ds_dir / spec.source_file
        if not (gt_path.exists() and result_path.exists() and pred_path.exists()):
            continue

        datasets_covered += 1
        gt_lines = gt_path.read_text(encoding="utf-8").splitlines()
        pred_lines = pred_path.read_text(encoding="utf-8").splitlines()

        pages_processed += len(gt_lines)
        for idx, gt_line in enumerate(gt_lines):
            gt = tokenize(gt_line)
            pred = tokenize(pred_lines[idx]) if idx < len(pred_lines) else []
            letters_total += len(gt)
            if any(t == "?" for t in pred):
                q_lines += 1
            for j, g in enumerate(gt):
                p = pred[j] if j < len(pred) else "?"
                if p == "?":
                    q_tokens += 1
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

    pos = tp + fn
    neg = tn + fp
    accuracy = (letters_correct / letters_total) if letters_total else 0.0
    recall = (tp / pos) if pos else 0.0
    fnr = (fn / pos) if pos else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    specificity = (tn / neg) if neg else 0.0
    class_acc = ((tp + tn) / (pos + neg)) if (pos + neg) else 0.0
    return {
        "datasets_covered": datasets_covered,
        "pages_processed": pages_processed,
        "letters_total": letters_total,
        "letters_correct": letters_correct,
        "letters_wrong": letters_total - letters_correct,
        "accuracy_percent": accuracy * 100.0,
        "extra_pred_tokens": extra_pred_tokens,
        "q_tokens": q_tokens,
        "q_lines": q_lines,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "fnr": fnr,
        "precision": precision,
        "specificity": specificity,
        "classification_accuracy": class_acc,
    }


def main() -> int:
    base = Path(__file__).resolve().parent
    dataset_root = base / "data" / "dataset"
    out_dir = base / "Results" / "publication_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[List[object]] = []
    for spec in MODELS:
        m = evaluate_model(dataset_root, spec)
        rows.append(
            [
                spec.name,
                spec.source_file,
                int(m["datasets_covered"]),
                int(m["pages_processed"]),
                int(m["letters_total"]),
                int(m["letters_correct"]),
                int(m["letters_wrong"]),
                f"{m['accuracy_percent']:.2f}",
                int(m["extra_pred_tokens"]),
                int(m["q_tokens"]),
                int(m["q_lines"]),
                int(m["tp"]),
                int(m["fn"]),
                int(m["fp"]),
                int(m["tn"]),
                f"{m['recall']:.6f}",
                f"{m['fnr']:.6f}",
                f"{m['precision']:.6f}",
                f"{m['specificity']:.6f}",
                f"{m['classification_accuracy']:.6f}",
            ]
        )

    out_csv = out_dir / "premium_subset_1_10_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "source_file",
                "datasets_covered",
                "pages_processed",
                "letters_total",
                "letters_correct",
                "letters_wrong",
                "accuracy_percent",
                "extra_pred_tokens",
                "q_tokens",
                "q_lines",
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

    rows_sorted = sorted(rows, key=lambda r: float(r[16]))  # FNR ascending
    out_md = out_dir / "premium_subset_1_10_metrics.md"
    lines = [
        "# Premium Subset Metrics (Datasets 1..10)",
        "",
        "| Model | Acc % | TP | FN | FP | TN | Recall | FNR | Precision | Specificity | Class-Acc | q_tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r[0]} | {r[7]} | {r[11]} | {r[12]} | {r[13]} | {r[14]} | {float(r[15]):.4f} | {float(r[16]):.4f} | {float(r[17]):.4f} | {float(r[18]):.4f} | {float(r[19]):.4f} | {r[9]} |"
        )

    by_model = {str(r[0]): r for r in rows}
    premium_focus = [
        "LLM OpenRouter Gemini-3.1 Flash-Lite-plain",
        "LLM OpenRouter Gemini-3.1 Flash-Lite-weak",
        "LLM Google Gemini-3.5 Flash-plain",
        "LLM Google Gemini-3.5 Flash-weak",
        "LLM OpenRouter OpenAI GPT-5.2-plain",
        "LLM OpenRouter OpenAI GPT-5.2-weak",
        "LLM OpenRouter OpenAI GPT-5.5-plain-subset_1_10",
        "LLM OpenRouter OpenAI GPT-5.5-weak-subset_1_10",
        "LLM OpenRouter Anthropic Claude Opus 4.7-plain-subset_1_10",
        "LLM OpenRouter Anthropic Claude Opus 4.7-weak-subset_1_10",
        "LLM OpenRouter xAI Grok-4.3-plain",
        "LLM OpenRouter xAI Grok-4.3-weak",
        "LLM OpenRouter Qwen3-VL-30B-plain",
        "LLM OpenRouter Qwen3-VL-30B-weak",
    ]
    lines += ["", "## Premium Models Only (Datasets 1..10)", "", "| Model | Acc % | Recall | FNR | Precision | Specificity |", "|---|---:|---:|---:|---:|---:|"]
    for name in premium_focus:
        r = by_model.get(name)
        if not r:
            continue
        model_label = name.replace("LLM OpenRouter ", "")
        lines.append(f"| {model_label} | {r[7]} | {float(r[15]):.4f} | {float(r[16]):.4f} | {float(r[17]):.4f} | {float(r[18]):.4f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
