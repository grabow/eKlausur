#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import random
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DATASET_IDS = list(range(1, 62))
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


@dataclass
class PageRow:
    dataset_id: int
    page_idx: int
    gt_tokens: List[str]
    pred_tokens: List[str]
    correct: int
    substitutions: int
    deletions: int
    insertions: int


def tokenize(line: str) -> List[str]:
    return [t.strip().upper() for t in line.strip().split() if t.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z / n)
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def binom_two_sided_p_value(k: int, n: int, p0: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    # Two-sided exact binomial p-value by probability ordering.
    observed = math.comb(n, k) * (p0**k) * ((1.0 - p0) ** (n - k))
    p = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (p0**i) * ((1.0 - p0) ** (n - i))
        if pi <= observed + 1e-15:
            p += pi
    return min(1.0, p)


def run_cmd(cmd: List[str]) -> str:
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return cp.stdout.strip()
    except Exception:
        return ""


def get_hardware_info() -> Dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split("\n")[0],
    }
    if sys.platform == "darwin":
        info["cpu_brand"] = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem_bytes = run_cmd(["sysctl", "-n", "hw.memsize"])
        info["memory_bytes"] = mem_bytes
        try:
            info["memory_gb"] = f"{int(mem_bytes) / (1024**3):.2f}"
        except Exception:
            info["memory_gb"] = ""
    return info


def load_summary(summary_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def load_prompt_text(llm_root: Path, prompt_index: int, expected_mode: str) -> str:
    prompts_path = llm_root / "prompts.py"
    if not prompts_path.exists():
        return ""
    spec = importlib.util.spec_from_file_location("llm_prompts_dyn", str(prompts_path))
    if spec is None or spec.loader is None:
        return ""
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if expected_mode == "studsolution_line":
        lst = getattr(mod, "EXPECTED_PROMPTS", [])
    else:
        lst = getattr(mod, "SINGLE_SHOT_PROMPTS", [])
    if not isinstance(lst, list):
        return ""
    if prompt_index < 0 or prompt_index >= len(lst):
        return ""
    return str(lst[prompt_index])


def load_openrouter_model_meta(model_id: str) -> Dict[str, object]:
    url = "https://openrouter.ai/api/v1/models"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for model in payload.get("data", []):
        if str(model.get("id", "")).strip() == model_id:
            return model
    return {}


def collect_page_rows(dataset_root: Path, model_dir: Path, source_file: str) -> List[PageRow]:
    rows: List[PageRow] = []
    for ds in DATASET_IDS:
        gt_path = dataset_root / str(ds) / "studSolution.txt"
        pred_path = model_dir / str(ds) / source_file
        if not gt_path.exists() or not pred_path.exists():
            raise FileNotFoundError(f"Missing files for dataset {ds}: {gt_path} / {pred_path}")

        gt_lines = gt_path.read_text(encoding="utf-8").splitlines()
        pred_lines = pred_path.read_text(encoding="utf-8").splitlines()
        for page_idx, gt_line in enumerate(gt_lines):
            gt = tokenize(gt_line)
            pred = tokenize(pred_lines[page_idx]) if page_idx < len(pred_lines) else []
            correct = 0
            substitutions = 0
            deletions = 0
            for i, gt_tok in enumerate(gt):
                if i >= len(pred):
                    deletions += 1
                    continue
                if pred[i] == gt_tok:
                    correct += 1
                else:
                    substitutions += 1
            insertions = max(0, len(pred) - len(gt))
            rows.append(
                PageRow(
                    dataset_id=ds,
                    page_idx=page_idx,
                    gt_tokens=gt,
                    pred_tokens=pred,
                    correct=correct,
                    substitutions=substitutions,
                    deletions=deletions,
                    insertions=insertions,
                )
            )
    return rows


def model_outcomes_for_mcnemar(dataset_root: Path, model_dir: Path, source_file: str) -> List[bool]:
    out: List[bool] = []
    for ds in DATASET_IDS:
        gt_lines = (dataset_root / str(ds) / "studSolution.txt").read_text(encoding="utf-8").splitlines()
        pred_lines = (model_dir / str(ds) / source_file).read_text(encoding="utf-8").splitlines()
        for page_idx, gt_line in enumerate(gt_lines):
            gt = tokenize(gt_line)
            pred = tokenize(pred_lines[page_idx]) if page_idx < len(pred_lines) else []
            for i, gt_tok in enumerate(gt):
                pred_tok = pred[i] if i < len(pred) else ""
                out.append(pred_tok == gt_tok)
    return out


def mcnemar(a: List[bool], b: List[bool]) -> Dict[str, float]:
    if len(a) != len(b):
        raise ValueError("Vectors must be same length for McNemar.")
    n10 = 0
    n01 = 0
    a_correct = 0
    b_correct = 0
    for av, bv in zip(a, b):
        if av:
            a_correct += 1
        if bv:
            b_correct += 1
        if av and (not bv):
            n10 += 1
        elif (not av) and bv:
            n01 += 1
    denom = n10 + n01
    if denom == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = ((abs(n10 - n01) - 1.0) ** 2) / denom
        p = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
    return {
        "n10": float(n10),
        "n01": float(n01),
        "chi2_cc": float(chi2),
        "p_value": float(p),
        "a_correct": float(a_correct),
        "b_correct": float(b_correct),
        "tokens": float(len(a)),
    }


def bootstrap_ci_for_dataset_delta(delta_by_dataset: List[float], b: int = 5000, seed: int = 7) -> Tuple[float, float]:
    rnd = random.Random(seed)
    n = len(delta_by_dataset)
    if n <= 1:
        d = delta_by_dataset[0] if delta_by_dataset else 0.0
        return (d, d)
    vals: List[float] = []
    for _ in range(b):
        sample = [delta_by_dataset[rnd.randrange(n)] for _ in range(n)]
        vals.append(sum(sample) / n)
    vals.sort()
    lo = vals[int(0.025 * (b - 1))]
    hi = vals[int(0.975 * (b - 1))]
    return (lo, hi)


def main() -> int:
    p = argparse.ArgumentParser(description="Build publication package for Gemini 3.1 Flash Lite run.")
    p.add_argument("--base-dir", default="/Users/wiggel/Python/eKlausur2")
    p.add_argument("--dataset-root", default="/Users/wiggel/Python/eKlausur2/data/dataset")
    p.add_argument(
        "--model-dir",
        default="/Users/wiggel/Python/eKlausur2/Results/LLM OpenRouter Gemini-3.1 Flash-Lite",
    )
    p.add_argument("--source-file", default="recognition_llm.txt")
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--provider-model", default="google/gemini-3.1-flash-lite")
    p.add_argument("--prompt-index", type=int, default=0)
    p.add_argument("--expected-mode", default="none", choices=["none", "studsolution_line"])
    p.add_argument("--llm-root", default="/Users/wiggel/Python/llm/llm")
    p.add_argument("--out-dir", default=None, help="Default: <model-dir>/publication_package")
    p.add_argument("--run-log", default=None, help="Optional recognizer log file to parse.")
    p.add_argument("--prompt-tokens", type=int, default=None, help="Optional total prompt tokens for exact cost.")
    p.add_argument(
        "--completion-tokens",
        type=int,
        default=None,
        help="Optional total completion tokens for exact cost.",
    )
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    llm_root = Path(args.llm_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (model_dir / "publication_package")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = model_dir / "_summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary = load_summary(summary_path)

    page_rows = collect_page_rows(dataset_root, model_dir, args.source_file)
    pages_processed = len(page_rows)
    letters_total = sum(len(r.gt_tokens) for r in page_rows)
    letters_correct = sum(r.correct for r in page_rows)
    substitutions = sum(r.substitutions for r in page_rows)
    deletions = sum(r.deletions for r in page_rows)
    insertions = sum(r.insertions for r in page_rows)
    letters_wrong = letters_total - letters_correct
    accuracy = (letters_correct / letters_total) if letters_total else 0.0
    ci_lo, ci_hi = wilson_ci(letters_correct, letters_total)

    # Per token + confusion
    token_rows: List[List[object]] = []
    confusion: Counter[Tuple[str, str]] = Counter()
    gt_class_stats: Dict[str, Dict[str, int]] = {c: {"support": 0, "correct": 0, "substitution": 0, "deletion": 0} for c in LETTERS}
    pred_letter_counts = Counter()
    pred_tp_counts = Counter()
    insertion_letter_counts = Counter()

    for r in page_rows:
        gt = r.gt_tokens
        pred = r.pred_tokens
        for i, gt_tok in enumerate(gt):
            pred_tok = pred[i] if i < len(pred) else ""
            if not pred_tok:
                outcome = "deletion"
                confusion[(gt_tok, "<DEL>")] += 1
                if gt_tok in gt_class_stats:
                    gt_class_stats[gt_tok]["support"] += 1
                    gt_class_stats[gt_tok]["deletion"] += 1
            elif pred_tok == gt_tok:
                outcome = "correct"
                confusion[(gt_tok, pred_tok)] += 1
                if gt_tok in gt_class_stats:
                    gt_class_stats[gt_tok]["support"] += 1
                    gt_class_stats[gt_tok]["correct"] += 1
                if pred_tok in LETTERS:
                    pred_tp_counts[pred_tok] += 1
                    pred_letter_counts[pred_tok] += 1
            else:
                outcome = "substitution"
                confusion[(gt_tok, pred_tok)] += 1
                if gt_tok in gt_class_stats:
                    gt_class_stats[gt_tok]["support"] += 1
                    gt_class_stats[gt_tok]["substitution"] += 1
                if pred_tok in LETTERS:
                    pred_letter_counts[pred_tok] += 1

            token_rows.append([r.dataset_id, r.page_idx, i, gt_tok, pred_tok if pred_tok else "<DEL>", outcome])

        if len(pred) > len(gt):
            for j in range(len(gt), len(pred)):
                tok = pred[j]
                token_rows.append([r.dataset_id, r.page_idx, j, "<NONE>", tok, "insertion"])
                insertion_letter_counts[tok] += 1
                confusion[("<INS>", tok)] += 1
                if tok in LETTERS:
                    pred_letter_counts[tok] += 1

    # Per-page CSV
    with (out_dir / "per_page_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset_id",
                "page_idx",
                "gt_tokens",
                "pred_tokens",
                "correct",
                "substitutions",
                "deletions",
                "insertions",
                "accuracy",
            ]
        )
        for r in page_rows:
            gt_n = len(r.gt_tokens)
            page_acc = (r.correct / gt_n) if gt_n else 0.0
            w.writerow(
                [
                    r.dataset_id,
                    r.page_idx,
                    gt_n,
                    len(r.pred_tokens),
                    r.correct,
                    r.substitutions,
                    r.deletions,
                    r.insertions,
                    f"{page_acc:.6f}",
                ]
            )

    with (out_dir / "per_token_outcomes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_id", "page_idx", "token_idx", "gt_token", "pred_token", "outcome"])
        w.writerows(token_rows)

    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_token", "pred_token", "count"])
        for (g, p_), c in sorted(confusion.items()):
            w.writerow([g, p_, c])

    # Per-class metrics
    with (out_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class",
                "support_gt",
                "correct_tp",
                "substitutions",
                "deletions",
                "recall",
                "predicted_total",
                "precision",
            ]
        )
        for c in LETTERS:
            support = gt_class_stats[c]["support"]
            tp = gt_class_stats[c]["correct"]
            sub = gt_class_stats[c]["substitution"]
            dele = gt_class_stats[c]["deletion"]
            pred_total = pred_letter_counts[c]
            recall = (tp / support) if support else 0.0
            precision = (tp / pred_total) if pred_total else 0.0
            w.writerow([c, support, tp, sub, dele, f"{recall:.6f}", pred_total, f"{precision:.6f}"])

    # Per-dataset summary for paired tests
    dataset_metrics: Dict[int, Dict[str, float]] = {}
    for ds in DATASET_IDS:
        ds_rows = [r for r in page_rows if r.dataset_id == ds]
        ds_total = sum(len(r.gt_tokens) for r in ds_rows)
        ds_correct = sum(r.correct for r in ds_rows)
        ds_acc = (ds_correct / ds_total) if ds_total else 0.0
        dataset_metrics[ds] = {"letters_total": ds_total, "letters_correct": ds_correct, "accuracy": ds_acc}

    # Statistical comparison vs YOLO models from Results/*/_summary.txt
    comparison_rows: List[List[object]] = []
    this_outcomes = model_outcomes_for_mcnemar(dataset_root, model_dir, args.source_file)
    results_root = base_dir / "Results"
    for other_summary in sorted(results_root.glob("*/_summary.txt")):
        if other_summary.parent.resolve() == model_dir.resolve():
            continue
        other_data = load_summary(other_summary)
        other_model_name = other_data.get("model", other_summary.parent.name)
        other_source = other_data.get("source_file", "")
        if not other_source:
            continue
        other_dir = other_summary.parent
        other_outcomes = model_outcomes_for_mcnemar(dataset_root, other_dir, other_source)
        mc = mcnemar(this_outcomes, other_outcomes)

        # Dataset-level paired deltas (this - other)
        deltas = []
        wins = 0
        losses = 0
        ties = 0
        for ds in DATASET_IDS:
            other_gt = (dataset_root / str(ds) / "studSolution.txt").read_text(encoding="utf-8").splitlines()
            other_pred = (other_dir / str(ds) / other_source).read_text(encoding="utf-8").splitlines()
            o_total = 0
            o_correct = 0
            for page_idx, gt_line in enumerate(other_gt):
                gt = tokenize(gt_line)
                pred = tokenize(other_pred[page_idx]) if page_idx < len(other_pred) else []
                o_total += len(gt)
                for i, gt_tok in enumerate(gt):
                    if i < len(pred) and pred[i] == gt_tok:
                        o_correct += 1
            o_acc = (o_correct / o_total) if o_total else 0.0
            d = dataset_metrics[ds]["accuracy"] - o_acc
            deltas.append(d)
            if d > 0:
                wins += 1
            elif d < 0:
                losses += 1
            else:
                ties += 1

        ci_d_lo, ci_d_hi = bootstrap_ci_for_dataset_delta(deltas, b=5000, seed=7)
        p_sign = binom_two_sided_p_value(wins, wins + losses, 0.5) if (wins + losses) > 0 else 1.0
        comparison_rows.append(
            [
                other_model_name,
                int(mc["tokens"]),
                int(mc["n10"]),
                int(mc["n01"]),
                f"{mc['chi2_cc']:.6f}",
                f"{mc['p_value']:.6e}",
                wins,
                losses,
                ties,
                f"{(sum(deltas) / len(deltas)) * 100.0:.4f}",
                f"{ci_d_lo * 100.0:.4f}",
                f"{ci_d_hi * 100.0:.4f}",
                f"{p_sign:.6e}",
            ]
        )

    with (out_dir / "statistical_comparison_vs_others.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "other_model",
                "tokens_compared",
                "n10_this_correct_other_wrong",
                "n01_this_wrong_other_correct",
                "mcnemar_chi2_cc",
                "mcnemar_p_value",
                "dataset_wins",
                "dataset_losses",
                "dataset_ties",
                "mean_delta_accuracy_pp_this_minus_other",
                "bootstrap95_low_pp",
                "bootstrap95_high_pp",
                "sign_test_p_value",
            ]
        )
        w.writerows(comparison_rows)

    # Reproducibility metadata
    prompt_text = load_prompt_text(llm_root, args.prompt_index, args.expected_mode)
    hardware = get_hardware_info()

    pred_files_in_results = [model_dir / str(i) / args.source_file for i in DATASET_IDS]
    pred_files_in_dataset = [dataset_root / str(i) / args.source_file for i in DATASET_IDS]
    mtimes = [p_.stat().st_mtime for p_ in pred_files_in_dataset if p_.exists()]
    mtime_source = "data/dataset"
    if len(mtimes) < 2:
        mtimes = [p_.stat().st_mtime for p_ in pred_files_in_results if p_.exists()]
        mtime_source = "Results model folder"
    approx_wallclock_s = 0.0
    if len(mtimes) >= 2:
        approx_wallclock_s = max(mtimes) - min(mtimes)

    avg_request_latency_s = None
    usage_prompt_tokens_total = None
    usage_completion_tokens_total = None
    usage_total_tokens_total = None
    if args.run_log:
        log_path = Path(args.run_log).expanduser().resolve()
        if log_path.exists():
            latencies = []
            p_tok = 0
            c_tok = 0
            t_tok = 0
            for line in log_path.read_text(encoding="utf-8").splitlines():
                m = line.strip()
                if "recognize response received" in m and "elapsed=" in m:
                    try:
                        part = m.split("elapsed=", 1)[1]
                        sec = part.split("s", 1)[0].strip()
                        latencies.append(float(sec))
                    except Exception:
                        pass
                if "usage: provider=openrouter" in m:
                    try:
                        fields = m.split("usage: provider=openrouter", 1)[1].strip().split()
                        kv = {}
                        for f in fields:
                            if "=" in f:
                                k, v = f.split("=", 1)
                                kv[k.strip()] = v.strip()
                        if kv.get("prompt_tokens", "-").isdigit():
                            p_tok += int(kv["prompt_tokens"])
                        if kv.get("completion_tokens", "-").isdigit():
                            c_tok += int(kv["completion_tokens"])
                        if kv.get("total_tokens", "-").isdigit():
                            t_tok += int(kv["total_tokens"])
                    except Exception:
                        pass
            if latencies:
                avg_request_latency_s = sum(latencies) / len(latencies)
            if p_tok > 0:
                usage_prompt_tokens_total = p_tok
            if c_tok > 0:
                usage_completion_tokens_total = c_tok
            if t_tok > 0:
                usage_total_tokens_total = t_tok

    model_meta = {}
    try:
        model_meta = load_openrouter_model_meta(args.provider_model)
    except Exception:
        model_meta = {}

    pricing = model_meta.get("pricing", {}) if isinstance(model_meta, dict) else {}
    prompt_price_per_token = float(str(pricing.get("prompt", "nan"))) if pricing else float("nan")
    completion_price_per_token = float(str(pricing.get("completion", "nan"))) if pricing else float("nan")

    prompt_tokens_for_cost = args.prompt_tokens if args.prompt_tokens is not None else usage_prompt_tokens_total
    completion_tokens_for_cost = args.completion_tokens if args.completion_tokens is not None else usage_completion_tokens_total
    cost_exact = None
    if (
        prompt_tokens_for_cost is not None
        and completion_tokens_for_cost is not None
        and not math.isnan(prompt_price_per_token)
        and not math.isnan(completion_price_per_token)
    ):
        cost_exact = prompt_tokens_for_cost * prompt_price_per_token + completion_tokens_for_cost * completion_price_per_token

    reproducibility = {
        "run_id": f"openrouter_gemini31flashlite_{summary.get('created_at', '').replace(':', '').replace('-', '')}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "provider_model": args.provider_model,
        "source_file": args.source_file,
        "dataset_root": str(dataset_root),
        "dataset_ids": DATASET_IDS,
        "prompt_index": args.prompt_index,
        "expected_mode": args.expected_mode,
        "prompt_text": prompt_text,
        "hyperparameters": {
            "temperature": None,
            "top_p": None,
            "seed": None,
            "response_format": "schema (with automatic fallback where unsupported)",
        },
        "preprocessing": {
            "location": "recognizer.copy_blurr_resize",
            "steps": [
                "grayscale read",
                "resize longest dimension to 1152 px",
                "cv2.INTER_LINEAR",
                "no gaussian blur applied (commented out)",
                "no inversion applied (commented out)",
            ],
        },
        "software": {
            "python": sys.version.split("\n")[0],
            "run_script": str(base_dir / "run_llm_recognition.py"),
            "llm_root": str(llm_root),
        },
        "hardware": hardware,
        "run_time_estimate_from_prediction_file_mtime": {
            "start_epoch_s": min(mtimes) if mtimes else None,
            "end_epoch_s": max(mtimes) if mtimes else None,
            "approx_wallclock_seconds": approx_wallclock_s,
            "throughput_pages_per_second": (pages_processed / approx_wallclock_s) if approx_wallclock_s > 0 else None,
            "mtime_source": mtime_source,
            "note": "approximation from prediction file mtimes; run log based timing is preferred",
        },
        "run_log": args.run_log,
        "avg_request_latency_seconds_from_log": avg_request_latency_s,
        "usage_from_log": {
            "prompt_tokens_total": usage_prompt_tokens_total,
            "completion_tokens_total": usage_completion_tokens_total,
            "total_tokens_total": usage_total_tokens_total,
        },
        "openrouter_model_metadata": model_meta,
        "costing": {
            "prompt_price_per_token_usd": None if math.isnan(prompt_price_per_token) else prompt_price_per_token,
            "completion_price_per_token_usd": None if math.isnan(completion_price_per_token) else completion_price_per_token,
            "prompt_tokens_total": prompt_tokens_for_cost,
            "completion_tokens_total": completion_tokens_for_cost,
            "estimated_total_cost_usd": cost_exact,
            "note": (
                "Either pass --prompt-tokens/--completion-tokens or provide --run-log "
                "with usage lines to compute exact cost. "
                "Without token counts, only price coefficients are recorded."
            ),
            "sources": {
                "openrouter_models_endpoint": "https://openrouter.ai/api/v1/models",
                "openrouter_chat_completion_usage_doc": "https://openrouter.ai/docs/api-reference/chat-completion",
            },
        },
    }
    (out_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Overall metric package
    metrics = {
        "pages_processed": pages_processed,
        "letters_total": letters_total,
        "letters_correct": letters_correct,
        "letters_wrong": letters_wrong,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "ci95_wilson_low": ci_lo,
        "ci95_wilson_high": ci_hi,
        "error_breakdown": {
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Robustness template
    robustness_template = {
        "purpose": "Repeat and sensitivity runs for stochastic/provider variability.",
        "recommended_repeats_same_setup": 3,
        "factors": [
            {"name": "prompt_index", "values": [0, 1], "type": "sensitivity"},
            {"name": "provider_model", "values": ["google/gemini-3.1-flash-lite", "google/gemini-3.1-flash-lite-preview"], "type": "sensitivity"},
            {"name": "expected_mode", "values": ["none", "studsolution_line"], "type": "sensitivity"},
        ],
        "run_command_template": (
            "/Users/wiggel/Python/eKlausur2/.venv/bin/python "
            "/Users/wiggel/Python/eKlausur2/run_llm_recognition.py "
            "--dataset-root /Users/wiggel/Python/eKlausur2/data/dataset "
            "--provider openrouter "
            "--provider-model {provider_model} "
            "--prompt-index {prompt_index} "
            "--expected-mode {expected_mode} "
            "--output-name {output_name}"
        ),
        "analysis_note": "Store each run under a separate model/result folder and rerun this package script per run.",
    }
    (out_dir / "robustness_plan.json").write_text(json.dumps(robustness_template, indent=2) + "\n", encoding="utf-8")

    readme_lines = [
        "# Gemini 3.1 Flash-Lite Publication Package",
        "",
        "Generated artifacts:",
        "- reproducibility.json",
        "- metrics_summary.json",
        "- per_page_metrics.csv",
        "- per_token_outcomes.csv",
        "- confusion_matrix.csv",
        "- per_class_metrics.csv",
        "- statistical_comparison_vs_others.csv",
        "- robustness_plan.json",
        "",
        "Notes:",
        "- Cost is exact only if prompt/completion token totals are provided to this script.",
        "- Timing is approximated from prediction file mtimes unless a recognizer run log is provided.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote Gemini package to: {out_dir}")
    print(f"[INFO] accuracy_percent={accuracy * 100.0:.2f} substitutions={substitutions} deletions={deletions} insertions={insertions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
