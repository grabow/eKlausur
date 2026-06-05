#!/usr/bin/env bash
set -u -o pipefail

ROOT="/Users/wiggel/Python/eKlausur2"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/run_llm_recognition.py"
DATASET_ROOT="$ROOT/data/dataset"

MODEL="gemini-3.5-flash"          # Google-direct (provider=gemini), nicht OpenRouter
OUT="recognition_llm_gemini35_flash_google_weak_hint_ref.txt"
START_DS="${START_DS:-1}"
END_DS="${END_DS:-10}"
MAX_DATASET_ATTEMPTS="${MAX_DATASET_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"

failed=0
for i in $(seq "$START_DS" "$END_DS"); do
  target="$DATASET_ROOT/$i/$OUT"
  if [[ -f "$target" ]]; then
    echo "[SKIP] $MODEL weak dataset $i -> exists"; continue
  fi
  ok=0
  for attempt in $(seq 1 "$MAX_DATASET_ATTEMPTS"); do
    echo "[RUN] $MODEL weak dataset $i (attempt $attempt/$MAX_DATASET_ATTEMPTS)"
    if "$PY" -u "$RUN" \
      --dataset-root "$DATASET_ROOT" \
      --dataset-id "$i" \
      --provider gemini \
      --provider-model "$MODEL" \
      --expected-mode reference_line \
      --output-name "$OUT"; then
      ok=1; break
    fi
    echo "[WARN] $MODEL weak dataset $i attempt $attempt failed."
    sleep "$RETRY_SLEEP_SECONDS"
  done
  # all-'?' = systemischer Ausfall (z.B. Timeout-Falle) -> verwerfen und stoppen
  if [[ -s "$target" ]] && ! grep -q '[A-Z]' "$target"; then
    echo "[ABORT] $MODEL weak dataset $i: Ausgabe komplett '?' (systemischer Ausfall / Timeout). Lauf gestoppt."
    rm -f "$target"; exit 3
  fi
  [[ "$ok" == "0" ]] && failed=$((failed + 1))
done
echo "[DONE] $MODEL weak failed_datasets=$failed"
