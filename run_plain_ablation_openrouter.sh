#!/usr/bin/env bash
set -u -o pipefail

ROOT="/Users/wiggel/Python/eKlausur2"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/run_llm_recognition.py"
DATASET_ROOT="$ROOT/data/dataset"

START_DS="${START_DS:-1}"
END_DS="${END_DS:-61}"
MAX_TOKENS="${OPENROUTER_MAX_TOKENS:-1024}"
RUN_GPT52="${RUN_GPT52:-1}"
RUN_QWEN30B="${RUN_QWEN30B:-1}"
RUN_GROK43="${RUN_GROK43:-1}"
MAX_DATASET_ATTEMPTS="${MAX_DATASET_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"

run_model() {
  local provider_model="$1"
  local out_file="$2"
  local failed=0
  for i in $(seq "$START_DS" "$END_DS"); do
    local target="$DATASET_ROOT/$i/$out_file"
    if [[ -f "$target" ]]; then
      echo "[SKIP] $provider_model dataset $i -> exists"
      continue
    fi
    local ok=0
    for attempt in $(seq 1 "$MAX_DATASET_ATTEMPTS"); do
      echo "[RUN] $provider_model dataset $i (attempt $attempt/$MAX_DATASET_ATTEMPTS)"
      if OPENROUTER_MAX_TOKENS="$MAX_TOKENS" "$PY" -u "$RUN" \
        --dataset-root "$DATASET_ROOT" \
        --dataset-id "$i" \
        --provider openrouter \
        --provider-model "$provider_model" \
        --expected-mode none \
        --output-name "$out_file"; then
        ok=1
        break
      fi
      echo "[WARN] $provider_model dataset $i attempt $attempt failed."
      sleep "$RETRY_SLEEP_SECONDS"
    done
    if [[ "$ok" == "0" ]]; then
      echo "[ERROR] $provider_model dataset $i failed after $MAX_DATASET_ATTEMPTS attempts; continue with next dataset."
      failed=$((failed + 1))
    fi
  done
  echo "[DONE] $provider_model failed_datasets=$failed"
}

if [[ "$RUN_GPT52" == "1" ]]; then
  run_model "openai/gpt-5.2" "recognition_llm_openrouter_openai_gpt52_plain.txt"
fi
if [[ "$RUN_QWEN30B" == "1" ]]; then
  run_model "qwen/qwen3-vl-30b-a3b-instruct" "recognition_llm_qwen3vl30b_plain.txt"
fi
if [[ "$RUN_GROK43" == "1" ]]; then
  run_model "x-ai/grok-4.3" "recognition_llm_grok43_plain.txt"
fi
