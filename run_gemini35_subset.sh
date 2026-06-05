#!/usr/bin/env bash
set -u -o pipefail

ROOT="/Users/wiggel/Python/eKlausur2"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/run_llm_recognition.py"
DATASET_ROOT="$ROOT/data/dataset"

MODEL="google/gemini-3.5-flash"
START_DS="${START_DS:-1}"
END_DS="${END_DS:-10}"
MAX_TOKENS="${OPENROUTER_MAX_TOKENS:-1024}"
MAX_DATASET_ATTEMPTS="${MAX_DATASET_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"

run_model() {
  local mode="$1"        # none | reference_line
  local out_file="$2"
  local failed=0

  # Pre-flight: HTTP 402 -> ganzer Lauf abgebrochen, bevor etwas geschrieben wird.
  "$PY" "$ROOT/check_openrouter_credit.py" \
    --model "$MODEL" --max-tokens "$MAX_TOKENS" \
    --image "$DATASET_ROOT/$START_DS/page_0.jpg" --env-file "$ROOT/.env"
  if [[ "$?" == "2" ]]; then
    echo "[ABORT] $MODEL ($mode): OpenRouter-Guthaben erschoepft (HTTP 402). Lauf gestoppt."
    exit 3
  fi

  for i in $(seq "$START_DS" "$END_DS"); do
    local target="$DATASET_ROOT/$i/$out_file"
    if [[ -f "$target" ]]; then
      echo "[SKIP] $MODEL $mode dataset $i -> exists"
      continue
    fi
    local ok=0
    for attempt in $(seq 1 "$MAX_DATASET_ATTEMPTS"); do
      echo "[RUN] $MODEL $mode dataset $i (attempt $attempt/$MAX_DATASET_ATTEMPTS)"
      if OPENROUTER_MAX_TOKENS="$MAX_TOKENS" "$PY" -u "$RUN" \
        --dataset-root "$DATASET_ROOT" \
        --dataset-id "$i" \
        --provider openrouter \
        --provider-model "$MODEL" \
        --expected-mode "$mode" \
        --output-name "$out_file"; then
        ok=1
        break
      fi
      echo "[WARN] $MODEL $mode dataset $i attempt $attempt failed."
      sleep "$RETRY_SLEEP_SECONDS"
    done
    # all-'?' = systemischer Ausfall -> verwerfen und stoppen
    if [[ -s "$target" ]] && ! grep -q '[A-Z]' "$target"; then
      echo "[ABORT] $MODEL $mode dataset $i: Ausgabe komplett '?' (systemischer Ausfall). Lauf gestoppt."
      rm -f "$target"
      exit 3
    fi
    if [[ "$ok" == "0" ]]; then
      echo "[ERROR] $MODEL $mode dataset $i failed after $MAX_DATASET_ATTEMPTS attempts; continue."
      failed=$((failed + 1))
    fi
  done
  echo "[DONE] $MODEL $mode failed_datasets=$failed"
}

run_model none           "recognition_llm_openrouter_gemini35_flash_plain.txt"
run_model reference_line "recognition_llm_openrouter_gemini35_flash_weak_hint_ref.txt"
