#!/usr/bin/env python3
"""
Run foundation-model recognition for each dataset folder.

Uses the same Python recognizer stack as eKlausur:
  /Users/wiggel/Python/llm/llm/recognizer.py

For each data/dataset/<id>/ with studSolution.txt:
- process page_*.jpg in numeric order
- call recognizer.copy_blurr_resize + recognizer.recognize(...)
- write one line per page to output file (space-separated letters)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple


PAGE_RE = re.compile(r"^page_(\d+)\.jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LLM-based recognition files per dataset.")
    p.add_argument("--dataset-root", default="data/dataset", help="Root folder containing numeric dataset folders.")
    p.add_argument(
        "--dataset-id",
        action="append",
        type=int,
        default=None,
        help="Optional dataset id filter. Can be provided multiple times (e.g. --dataset-id 1 --dataset-id 2).",
    )
    p.add_argument("--llm-root", default="/Users/wiggel/Python/llm/llm", help="Folder containing recognizer.py.")
    p.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini", "openrouter", "academiccloud", "ollama"],
        help="Recognizer provider backend.",
    )
    p.add_argument(
        "--provider-model",
        default=None,
        help=(
            "Optional concrete model name override for the selected provider "
            "(e.g. gpt-5.2-2025-12-11, gemini-2.5-flash)."
        ),
    )
    p.add_argument("--prompt-index", type=int, default=0, help="Prompt index passed to recognizer.recognize(...).")
    p.add_argument(
        "--expected-mode",
        choices=["none", "studsolution_line"],
        default="none",
        help="Optional expected-hint mode passed to recognizer.",
    )
    p.add_argument("--output-name", default="recognition_llm.txt", help="Output filename per dataset folder.")
    p.add_argument("--fail-token", default="?", help="Token used when page-level recognition fails.")
    p.add_argument(
        "--raw-json-dir",
        default=None,
        help="Optional directory where raw recognizer JSON responses are stored (<dataset>/<page>.json).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional recognizer log file path (forwarded via recognizer.set_log_file).",
    )
    p.add_argument(
        "--env-file",
        default=".env",
        help="Local env file for API keys (default: .env). Required.",
    )
    return p.parse_args()


def is_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_stud = (path / "studSolution.txt").exists()
    has_page = any(PAGE_RE.match(p.name) for p in path.iterdir() if p.is_file())
    return has_stud and has_page


def iter_dataset_dirs(dataset_root: Path, dataset_ids: set[int] | None) -> Iterable[Path]:
    if is_dataset_dir(dataset_root):
        root_id = int(dataset_root.name) if dataset_root.name.isdigit() else None
        if dataset_ids is None or (root_id is not None and root_id in dataset_ids):
            yield dataset_root
        return

    dirs = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit() and is_dataset_dir(d)]
    for d in sorted(dirs, key=lambda x: int(x.name)):
        if dataset_ids is not None and int(d.name) not in dataset_ids:
            continue
        yield d


def list_page_images(dataset_dir: Path) -> List[Path]:
    pages: List[Tuple[int, Path]] = []
    for path in dataset_dir.iterdir():
        m = PAGE_RE.match(path.name)
        if not m:
            continue
        pages.append((int(m.group(1)), path))
    pages.sort(key=lambda t: t[0])
    return [p for _, p in pages]


def normalize_letter(raw: str | None, fail_token: str) -> str:
    if not raw:
        return fail_token
    value = raw.strip()
    if value == "?":
        return value
    if len(value) == 1 and value.isalpha():
        return value.upper()
    return fail_token


def parse_items_json(res_json: str, fail_token: str) -> List[str]:
    """
    Parse recognizer output JSON:
      {"items":[{"digit":{"digit":"1"...},"letter":{"letter":"A"...}}, ...], ...}
    """
    data = json.loads(res_json)
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        return [fail_token]

    parsed: List[Tuple[int, int, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        digit_obj = item.get("digit", {})
        letter_obj = item.get("letter", {})
        raw_letter = letter_obj.get("letter") if isinstance(letter_obj, dict) else None
        token = normalize_letter(raw_letter if isinstance(raw_letter, str) else None, fail_token)

        dval = None
        if isinstance(digit_obj, dict):
            raw_digit = digit_obj.get("digit")
            try:
                dval = int(str(raw_digit))
            except Exception:
                dval = None
        parsed.append((dval if dval is not None else 10_000 + idx, idx, token))

    if not parsed:
        return [fail_token]

    parsed.sort(key=lambda t: (t[0], t[1]))
    letters = [tok for _, _, tok in parsed]
    return letters if letters else [fail_token]


def load_env_file(env_path: Path) -> int:
    """
    Load KEY=VALUE pairs into os.environ.
    Supports optional leading 'export ' and quoted values.
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Required env file not found: {env_path}")
    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def apply_provider_model_override(recognizer, provider: str, model: str) -> None:
    provider = provider.strip().lower()
    model = model.strip()
    if not model:
        return

    if provider == "openai":
        recognizer.OPENAI_MODEL = model
        recognizer.RecognizerHelper.configure(OPENAI_MODEL=model)
    elif provider == "gemini":
        recognizer.MODEL_NAME = model
    elif provider == "openrouter":
        recognizer.OPENROUTER_MODEL = model
        recognizer.RecognizerHelper.configure(OPENROUTER_MODEL=model)
    elif provider == "academiccloud":
        recognizer.ACADEMICCLOUD_MODEL = model
        recognizer.RecognizerHelper.configure(ACADEMICCLOUD_MODEL=model)
    elif provider == "ollama":
        recognizer.OLLAMA_MODEL = model
        recognizer.RecognizerHelper.configure(OLLAMA_MODEL=model)

    # Also set ENV as fallback for code paths that prefer env variables.
    env_key = {
        "openai": "OPENAI_MODEL",
        "gemini": "MODEL_NAME",
        "openrouter": "OPENROUTER_MODEL",
        "academiccloud": "ACADEMICCLOUD_MODEL",
        "ollama": "OLLAMA_MODEL",
    }.get(provider)
    if env_key:
        os.environ[env_key] = model


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    llm_root = Path(args.llm_root).expanduser().resolve()
    raw_json_root = Path(args.raw_json_dir).expanduser().resolve() if args.raw_json_dir else None
    env_file = Path(args.env_file).expanduser().resolve() if args.env_file else None
    dataset_ids = set(args.dataset_id) if args.dataset_id else None

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not llm_root.exists():
        raise FileNotFoundError(f"LLM root not found: {llm_root}")

    if env_file is not None:
        loaded_env = load_env_file(env_file)
        if loaded_env <= 0:
            raise RuntimeError(f"Env file is empty or invalid: {env_file}")
        print(f"[INFO] Loaded {loaded_env} entries from env file: {env_file}")

    sys.path.insert(0, str(llm_root))
    import recognizer  # type: ignore
    # Force env-only key loading in this project: disable credentials.txt fallback.
    recognizer.OPENAI_CREDENTIALS_PATH = "__DISABLED__"
    recognizer.RecognizerHelper.configure(OPENAI_CREDENTIALS_PATH="__DISABLED__")

    if args.log_file:
        recognizer.set_log_file(str(Path(args.log_file).expanduser().resolve()))

    if args.provider_model:
        apply_provider_model_override(recognizer, args.provider, args.provider_model)
        print(f"[INFO] Provider model override: provider={args.provider} model={args.provider_model}")
    else:
        print(f"[INFO] Provider without explicit model override: {args.provider}")

    processed_datasets = 0
    processed_pages = 0
    failed_pages = 0

    for dataset_dir in iter_dataset_dirs(dataset_root, dataset_ids):
        stud_solution_path = dataset_dir / "studSolution.txt"
        if not stud_solution_path.exists():
            continue

        page_images = list_page_images(dataset_dir)
        if not page_images:
            continue

        stud_lines: List[str] = []
        if args.expected_mode == "studsolution_line":
            stud_lines = stud_solution_path.read_text(encoding="utf-8").splitlines()

        out_lines: List[str] = []
        for page_path in page_images:
            try:
                m = PAGE_RE.match(page_path.name)
                page_idx = int(m.group(1)) if m else 0
                expected = None
                if args.expected_mode == "studsolution_line" and page_idx < len(stud_lines):
                    expected = stud_lines[page_idx]

                with tempfile.TemporaryDirectory(prefix="llm_page_") as tmp_dir_str:
                    tmp_dir = Path(tmp_dir_str)
                    temp_subdir = tmp_dir / "Temp"
                    temp_subdir.mkdir(parents=True, exist_ok=True)
                    src_temp = temp_subdir / "page_0.jpeg"
                    shutil.copyfile(page_path, src_temp)

                    preproc_path = recognizer.copy_blurr_resize(str(tmp_dir), "Temp", "page_0.jpeg")
                    res_json = recognizer.recognize(
                        preproc_path,
                        expected=expected,
                        model=args.provider,
                        prompt=args.prompt_index,
                    )
                letters = parse_items_json(res_json, args.fail_token)
                out_lines.append(" ".join(letters))

                if raw_json_root is not None:
                    raw_out = raw_json_root / dataset_dir.name / f"{page_path.stem}.json"
                    raw_out.parent.mkdir(parents=True, exist_ok=True)
                    raw_out.write_text(res_json, encoding="utf-8")
            except Exception as ex:
                failed_pages += 1
                out_lines.append(args.fail_token)
                print(f"[WARN] {dataset_dir.name}/{page_path.name}: {ex}")
            processed_pages += 1

        out_file = dataset_dir / args.output_name
        out_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        processed_datasets += 1
        print(f"[OK] {dataset_dir.name}: wrote {out_file.name} ({len(out_lines)} lines)")

    print("=== Summary ===")
    print(f"Provider: {args.provider}")
    if args.provider_model:
        print(f"Provider model override: {args.provider_model}")
    print(f"Datasets processed: {processed_datasets}")
    print(f"Pages processed: {processed_pages}")
    print(f"Pages failed (fallback '{args.fail_token}'): {failed_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
