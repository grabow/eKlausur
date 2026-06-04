#!/usr/bin/env bash
set -u -o pipefail

ROOT="/Users/wiggel/Python/eKlausur2"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/run_llm_recognition.py"
DATASET_ROOT="$ROOT/data/dataset"

START_DS="${START_DS:-1}"
END_DS="${END_DS:-61}"
MAX_TOKENS="${OPENROUTER_MAX_TOKENS:-1024}"
MAX_DATASET_ATTEMPTS="${MAX_DATASET_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_GROK="${RUN_GROK:-1}"

run_model() {
  local provider_model="$1"
  local out_file="$2"
  local failed=0

  # Pre-flight: ein minimaler Vision-Probe. HTTP 402 -> ganzer Lauf abgebrochen,
  # bevor irgendetwas geschrieben wird.
  "$PY" "$ROOT/check_openrouter_credit.py" \
    --model "$provider_model" --max-tokens "$MAX_TOKENS" \
    --image "$DATASET_ROOT/$START_DS/page_0.jpg" --env-file "$ROOT/.env"
  local cr=$?
  if [[ "$cr" == "2" ]]; then
    echo "[ABORT] $provider_model: OpenRouter-Guthaben erschoepft (HTTP 402). Lauf gestoppt, keine Datensaetze geschrieben."
    exit 3
  fi

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
    # Abbruchkriterium: Ausgabe komplett '?' (kein A-Z) = systemischer Ausfall
    # (z.B. Guthaben mitten im Lauf erschoepft). Datei verwerfen und stoppen.
    if [[ -s "$target" ]] && ! grep -q '[A-Z]' "$target"; then
      echo "[ABORT] $provider_model dataset $i: Ausgabe komplett '?' (systemischer Ausfall, evtl. Guthaben erschoepft). Lauf gestoppt."
      rm -f "$target"
      exit 3
    fi
    if [[ "$ok" == "0" ]]; then
      echo "[ERROR] $provider_model dataset $i failed after $MAX_DATASET_ATTEMPTS attempts; continue."
      failed=$((failed + 1))
    fi
  done
  echo "[DONE] $provider_model failed_datasets=$failed"
}

# Qwen3-VL-30B: WICHTIG instruct-Variante (base-Variante lieferte 100% '?').
if [[ "$RUN_QWEN" == "1" ]]; then
  run_model "qwen/qwen3-vl-30b-a3b-instruct" "recognition_llm_qwen3vl30b_plain.txt"
fi
# Grok-4.3: 1-38 bereits vorhanden, fuellt 39-61.
if [[ "$RUN_GROK" == "1" ]]; then
  run_model "x-ai/grok-4.3" "recognition_llm_grok43_plain.txt"
fi
