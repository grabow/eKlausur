#!/usr/bin/env python3
"""Pre-flight credit/affordability probe for OpenRouter vision runs.

Sends one minimal vision request (system + short text + one image) with the same
max_tokens the real run uses, and inspects the HTTP status.

Exit codes:
  0  -> request affordable (HTTP 200) - safe to start the run
  2  -> HTTP 402 (out of credits / max_tokens too high) - caller should ABORT
  1  -> any other error (network, auth, unexpected) - caller decides

Usage:
  check_openrouter_credit.py --model qwen/qwen3-vl-30b-a3b-instruct \
      --max-tokens 1024 --image data/dataset/1/page_0.jpg --env-file .env
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


def load_key(env_file: str) -> str | None:
    try:
        for line in open(env_file, encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.strip().split("=", 1)[1].strip().strip('"')
    except FileNotFoundError:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--image", required=True)
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    key = load_key(args.env_file)
    if not key:
        print(f"[credit-check] OPENROUTER_API_KEY nicht in {args.env_file} gefunden.")
        return 1

    try:
        b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    except OSError as e:
        print(f"[credit-check] Probe-Bild nicht lesbar: {e}")
        return 1
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": "You are an OCR assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with the single word OK."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=args.timeout)
        r.read()
        print(f"[credit-check] OK (HTTP {r.status}) model={args.model} max_tokens={args.max_tokens}")
        return 0
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        if e.code == 402:
            print(f"[credit-check] HTTP 402 - Guthaben erschoepft / max_tokens zu hoch: {body}")
            return 2
        print(f"[credit-check] HTTP {e.code}: {body}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[credit-check] Fehler: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
