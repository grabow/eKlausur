#!/usr/bin/env bash
set -u -o pipefail

ROOT="/Users/wiggel/Python/eKlausur2"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/run_llm_recognition.py"
DATASET_ROOT="$ROOT/data/dataset"

MODEL="gemini-3.5-flash"          # Google-direct (provider=gemini)
START_DS="${START_DS:-1}"
END_DS="${END_DS:-61}"
MAX_DATASET_ATTEMPTS="${MAX_DATASET_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"

run_mode() {
  local mode="$1"        # none | reference_line
  local out_file="$2"
  local failed=0
  for i in $(seq "$START_DS" "$END_DS"); do
    local target="$DATASET_ROOT/$i/$out_file"
    if [[ -f "$target" ]]; then
      echo "[SKIP] $MODEL $mode dataset $i -> exists"; continue
    fi
    local ok=0
    for attempt in $(seq 1 "$MAX_DATASET_ATTEMPTS"); do
      echo "[RUN] $MODEL $mode dataset $i (attempt $attempt/$MAX_DATASET_ATTEMPTS)"
      if "$PY" -u "$RUN" \
        --dataset-root "$DATASET_ROOT" \
        --dataset-id "$i" \
        --provider gemini \
        --provider-model "$MODEL" \
        --expected-mode "$mode" \
        --output-name "$out_file"; then
        ok=1; break
      fi
      echo "[WARN] $MODEL $mode dataset $i attempt $attempt failed."
      sleep "$RETRY_SLEEP_SECONDS"
    done
    # all-'?' = systemischer Ausfall (z.B. Timeout-Falle) -> verwerfen und stoppen
    if [[ -s "$target" ]] && ! grep -q '[A-Z]' "$target"; then
      echo "[ABORT] $MODEL $mode dataset $i: Ausgabe komplett '?' (systemischer Ausfall / Timeout). Lauf gestoppt."
      rm -f "$target"; exit 3
    fi
    [[ "$ok" == "0" ]] && failed=$((failed + 1))
  done
  echo "[DONE] $MODEL $mode failed_datasets=$failed"
}

run_mode none           "recognition_llm_gemini35_flash_google_plain.txt"
run_mode reference_line "recognition_llm_gemini35_flash_google_weak_hint_ref.txt"
