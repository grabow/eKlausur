#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DATASET_IDS = list(range(1, 62))


@dataclass
class ModelConfig:
    name: str
    model_dir: Path
    source_file: str


@dataclass
class DatasetMetric:
    model: str
    dataset_id: int
    pages: int
    letters_total: int
    letters_correct: int
    letters_wrong: int
    extra_pred_tokens: int
    accuracy: float


@dataclass
class TokenOutcome:
    model: str
    correct: bool


@dataclass
class PageMetric:
    model: str
    dataset_id: int
    page_index: int
    tokens_total: int
    tokens_correct: int
    tokens_wrong: int
    extra_pred_tokens: int
    substitutions: int
    deletions: int
    insertions: int
    accuracy: float


def parse_summary(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def load_models(results_root: Path) -> List[ModelConfig]:
    models: List[ModelConfig] = []
    for summary_path in sorted(results_root.glob("*/_summary.txt")):
        data = parse_summary(summary_path)
        name = data.get("model")
        source_file = data.get("source_file")
        datasets_copied = data.get("datasets_copied")
        if not name or not source_file:
            continue
        try:
            copied = int(datasets_copied) if datasets_copied is not None else 0
        except ValueError:
            copied = 0
        # Publication full comparison expects complete 1..61 coverage.
        # Partial models (e.g. premium subset 1..10) are handled by separate subset reports.
        if copied < len(DATASET_IDS):
            continue
        models.append(ModelConfig(name=name, model_dir=summary_path.parent, source_file=source_file))
    if not models:
        raise RuntimeError(f"No model summaries found below: {results_root}")
    return models


def tokenize_line(line: str) -> List[str]:
    return [t.strip() for t in line.strip().split() if t.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z / n)
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def chi_square_sf_df1(x: float) -> float:
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def evaluate_model_dataset(
    dataset_root: Path,
    model: ModelConfig,
    dataset_id: int,
) -> Tuple[DatasetMetric, List[TokenOutcome], List[PageMetric], Dict[str, int]]:
    ds_dir = dataset_root / str(dataset_id)
    gt_path = ds_dir / "studSolution.txt"
    pred_path = model.model_dir / str(dataset_id) / model.source_file
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT file: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    gt_lines = gt_path.read_text(encoding="utf-8").splitlines()
    pred_lines = pred_path.read_text(encoding="utf-8").splitlines()

    letters_total = 0
    letters_correct = 0
    extra_pred_tokens = 0
    outcomes: List[TokenOutcome] = []
    page_rows: List[PageMetric] = []
    substitutions = 0
    deletions = 0
    insertions = 0

    for page_idx, gt_line in enumerate(gt_lines):
        gt_tokens = tokenize_line(gt_line)
        pred_tokens = tokenize_line(pred_lines[page_idx]) if page_idx < len(pred_lines) else []

        x = len(gt_tokens)
        page_correct = 0
        page_sub = 0
        page_del = 0
        letters_total += x
        for token_idx in range(x):
            gt = gt_tokens[token_idx].upper()
            if token_idx < len(pred_tokens):
                pred = pred_tokens[token_idx].upper()
            else:
                pred = "?"
                page_del += 1
            ok = pred == gt
            outcomes.append(TokenOutcome(model=model.name, correct=ok))
            if ok:
                letters_correct += 1
                page_correct += 1
            elif token_idx < len(pred_tokens):
                page_sub += 1
        if len(pred_tokens) > x:
            extra_pred_tokens += (len(pred_tokens) - x)
            insertions += (len(pred_tokens) - x)

        substitutions += page_sub
        deletions += page_del
        page_tokens_wrong = x - page_correct
        page_acc = (page_correct / x) if x else 0.0
        page_rows.append(
            PageMetric(
                model=model.name,
                dataset_id=dataset_id,
                page_index=page_idx + 1,
                tokens_total=x,
                tokens_correct=page_correct,
                tokens_wrong=page_tokens_wrong,
                extra_pred_tokens=max(0, len(pred_tokens) - x),
                substitutions=page_sub,
                deletions=page_del,
                insertions=max(0, len(pred_tokens) - x),
                accuracy=page_acc,
            )
        )

    letters_wrong = letters_total - letters_correct
    accuracy = (letters_correct / letters_total) if letters_total else 0.0
    metric = DatasetMetric(
        model=model.name,
        dataset_id=dataset_id,
        pages=len(gt_lines),
        letters_total=letters_total,
        letters_correct=letters_correct,
        letters_wrong=letters_wrong,
        extra_pred_tokens=extra_pred_tokens,
        accuracy=accuracy,
    )
    error_totals = {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
    return metric, outcomes, page_rows, error_totals


def aggregate_metrics(metrics: Iterable[DatasetMetric]) -> Dict[str, float]:
    rows = list(metrics)
    letters_total = sum(r.letters_total for r in rows)
    letters_correct = sum(r.letters_correct for r in rows)
    letters_wrong = sum(r.letters_wrong for r in rows)
    pages = sum(r.pages for r in rows)
    extra = sum(r.extra_pred_tokens for r in rows)
    accuracy = (letters_correct / letters_total) if letters_total else 0.0
    ci_lo, ci_hi = wilson_ci(letters_correct, letters_total)
    return {
        "datasets_copied": len(rows),
        "pages_processed": pages,
        "letters_total": letters_total,
        "letters_correct": letters_correct,
        "letters_wrong": letters_wrong,
        "extra_pred_tokens": extra,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
    }


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    base = Path(__file__).resolve().parent
    dataset_root = base / "data" / "dataset"
    results_root = base / "Results"
    out_dir = results_root / "publication_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(results_root)

    per_dataset_rows: List[List[object]] = []
    per_page_rows: List[List[object]] = []
    error_type_rows: List[List[object]] = []
    latency_rows: List[List[object]] = []
    overall_rows: List[List[object]] = []
    model_outcomes: Dict[str, List[bool]] = {m.name: [] for m in models}

    by_model_metrics: Dict[str, List[DatasetMetric]] = {m.name: [] for m in models}
    by_model_error_totals: Dict[str, Dict[str, int]] = {
        m.name: {"substitutions": 0, "deletions": 0, "insertions": 0} for m in models
    }

    for model in models:
        model_file_ts: List[Tuple[int, float]] = []
        for ds_id in DATASET_IDS:
            metric, outcomes, page_metrics, err_totals = evaluate_model_dataset(dataset_root, model, ds_id)
            by_model_metrics[model.name].append(metric)
            model_outcomes[model.name].extend([o.correct for o in outcomes])
            by_model_error_totals[model.name]["substitutions"] += err_totals["substitutions"]
            by_model_error_totals[model.name]["deletions"] += err_totals["deletions"]
            by_model_error_totals[model.name]["insertions"] += err_totals["insertions"]
            per_dataset_rows.append(
                [
                    model.name,
                    ds_id,
                    metric.pages,
                    metric.letters_total,
                    metric.letters_correct,
                    metric.letters_wrong,
                    metric.extra_pred_tokens,
                    f"{metric.accuracy:.6f}",
                    f"{metric.accuracy * 100.0:.2f}",
                ]
            )
            for pm in page_metrics:
                per_page_rows.append(
                    [
                        pm.model,
                        pm.dataset_id,
                        pm.page_index,
                        pm.tokens_total,
                        pm.tokens_correct,
                        pm.tokens_wrong,
                        pm.extra_pred_tokens,
                        pm.substitutions,
                        pm.deletions,
                        pm.insertions,
                        f"{pm.accuracy:.6f}",
                        f"{pm.accuracy * 100.0:.2f}",
                    ]
                )

            # Prefer dataset output file mtime for duration approximation.
            # Fallback to copied Results file if dataset file is unavailable.
            dataset_pred_path = dataset_root / str(ds_id) / model.source_file
            result_pred_path = model.model_dir / str(ds_id) / model.source_file
            ts_path = dataset_pred_path if dataset_pred_path.exists() else result_pred_path
            if ts_path.exists():
                model_file_ts.append((ds_id, ts_path.stat().st_mtime))

        agg = aggregate_metrics(by_model_metrics[model.name])
        overall_rows.append(
            [
                model.name,
                model.source_file,
                int(agg["datasets_copied"]),
                int(agg["pages_processed"]),
                int(agg["letters_total"]),
                int(agg["letters_correct"]),
                int(agg["letters_wrong"]),
                int(agg["extra_pred_tokens"]),
                f"{agg['accuracy']:.6f}",
                f"{agg['accuracy_percent']:.2f}",
                f"{agg['ci95_low']:.6f}",
                f"{agg['ci95_high']:.6f}",
            ]
        )
        error_type_rows.append(
            [
                model.name,
                by_model_error_totals[model.name]["substitutions"],
                by_model_error_totals[model.name]["deletions"],
                by_model_error_totals[model.name]["insertions"],
            ]
        )

        # Dataset-level duration approximation from result file mtimes.
        model_file_ts.sort(key=lambda t: t[0])
        prev_ts = None
        for ds_id, end_ts in model_file_ts:
            start_ts = prev_ts if prev_ts is not None else end_ts
            duration = max(0.0, end_ts - start_ts)
            latency_rows.append(
                [
                    model.name,
                    ds_id,
                    datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
                    datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
                    f"{duration:.3f}",
                ]
            )
            prev_ts = end_ts

    write_csv(
        out_dir / "model_per_dataset.csv",
        [
            "model",
            "dataset_id",
            "pages",
            "letters_total",
            "letters_correct",
            "letters_wrong",
            "extra_pred_tokens",
            "accuracy",
            "accuracy_percent",
        ],
        per_dataset_rows,
    )

    write_csv(
        out_dir / "model_per_page.csv",
        [
            "model",
            "dataset_id",
            "page_index",
            "tokens_total",
            "tokens_correct",
            "tokens_wrong",
            "extra_pred_tokens",
            "substitutions",
            "deletions",
            "insertions",
            "accuracy",
            "accuracy_percent",
        ],
        per_page_rows,
    )

    write_csv(
        out_dir / "model_error_types.csv",
        [
            "model",
            "substitutions",
            "deletions",
            "insertions",
        ],
        error_type_rows,
    )

    write_csv(
        out_dir / "model_latency_dataset.csv",
        [
            "model",
            "dataset_id",
            "start_ts_utc",
            "end_ts_utc",
            "duration_seconds",
        ],
        latency_rows,
    )

    write_csv(
        out_dir / "model_overall.csv",
        [
            "model",
            "source_file",
            "datasets_copied",
            "pages_processed",
            "letters_total",
            "letters_correct",
            "letters_wrong",
            "extra_pred_tokens",
            "accuracy",
            "accuracy_percent",
            "ci95_low",
            "ci95_high",
        ],
        overall_rows,
    )

    pair_rows: List[List[object]] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = models[i].name
            b = models[j].name
            a_vec = model_outcomes[a]
            b_vec = model_outcomes[b]
            if len(a_vec) != len(b_vec):
                raise RuntimeError(f"Token count mismatch for pair {a} vs {b}: {len(a_vec)} vs {len(b_vec)}")

            n10 = 0  # a correct, b wrong
            n01 = 0  # a wrong, b correct
            a_correct = 0
            b_correct = 0
            for av, bv in zip(a_vec, b_vec):
                if av:
                    a_correct += 1
                if bv:
                    b_correct += 1
                if av and (not bv):
                    n10 += 1
                elif (not av) and bv:
                    n01 += 1

            denom = n01 + n10
            if denom == 0:
                chi2 = 0.0
                p = 1.0
            else:
                chi2 = ((abs(n01 - n10) - 1.0) ** 2) / denom
                p = chi_square_sf_df1(chi2)

            delta_acc_pp = ((b_correct - a_correct) / len(a_vec)) * 100.0
            pair_rows.append(
                [
                    a,
                    b,
                    len(a_vec),
                    a_correct,
                    b_correct,
                    n10,
                    n01,
                    f"{chi2:.6f}",
                    f"{p:.6e}",
                    f"{delta_acc_pp:.4f}",
                ]
            )

    write_csv(
        out_dir / "pairwise_mcnemar.csv",
        [
            "model_a",
            "model_b",
            "tokens_compared",
            "model_a_correct",
            "model_b_correct",
            "n10_a_correct_b_wrong",
            "n01_a_wrong_b_correct",
            "mcnemar_chi2_cc",
            "p_value",
            "delta_accuracy_pp_b_minus_a",
        ],
        pair_rows,
    )

    overall_sorted = sorted(overall_rows, key=lambda r: float(r[9]), reverse=True)
    lines = [
        "# Publication Comparison (YOLO vs LLM)",
        "",
        f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overall Accuracy",
        "",
        "| Model | Accuracy % | 95% CI (Wilson) | Correct / Total | Extra Tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall_sorted:
        model = str(row[0])
        acc_pct = str(row[9])
        ci = f"{float(row[10])*100.0:.2f}-{float(row[11])*100.0:.2f}"
        corr = f"{row[5]} / {row[4]}"
        extra = str(row[7])
        lines.append(f"| {model} | {acc_pct} | {ci} | {corr} | {extra} |")

    lines += [
        "",
        "## Pairwise Significance (McNemar, continuity-corrected)",
        "",
        "| Model A | Model B | n10 (A right/B wrong) | n01 (A wrong/B right) | p-value | Delta pp (B-A) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[5]} | {row[6]} | {row[8]} | {row[9]} |")

    lines += [
        "",
        "## Files",
        "",
        "- `model_overall.csv`",
        "- `model_per_dataset.csv`",
        "- `model_per_page.csv`",
        "- `model_error_types.csv`",
        "- `model_latency_dataset.csv`",
        "- `pairwise_mcnemar.csv`",
    ]

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote publication comparison artifacts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
