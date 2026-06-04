# Results

## Scope

Vergleich der Inferenz auf `data/dataset/1..61` (insgesamt 742 Seiten, 3141 Tokens) mit:
- YOLOv5 Medium (Seed-Modell)
- YOLO26 Medium (Modell `yolo26m_20260506_best.pt`)

## YOLOv5 Medium

- `letters_total`: 3141
- `letters_correct`: 2856
- `letters_wrong`: 285
- `accuracy_percent`: 90.93
- `extra_pred_tokens`: 24

Quelle:
- `/Users/wiggel/Python/eKlausur2/Results/YOLOv5 Medium/_summary.txt`

## YOLO26 Medium (nach Bugfix)

- `letters_total`: 3141
- `letters_correct`: 2801
- `letters_wrong`: 340
- `accuracy_percent`: 89.18
- `extra_pred_tokens`: 6

Quelle:
- `/Users/wiggel/Python/eKlausur2/Results/YOLO26 Medium/_summary.txt`

## Root Cause Analysis (YOLO26)

Verbleibender Gap zu YOLOv5:
- YOLO26 liegt bei 89.18% vs. YOLOv5 bei 90.93%.
- Fehlerbild YOLO26 ist stark von `?`-Ausgaben geprägt.
- Haupttreiber: `x` (crossed_out) wird häufig als bester Kandidat pro Digit gewählt und anschließend zu `?` gemappt.

Messwerte aus Ablation:
- Baseline (aktuell): `89.18%` (`2801/3141`)
- `prefer_non_x` (Analysemodus): `90.80%` (`2852/3141`)
- Zugewinn: `+51` korrekte Tokens

Interpretation:
- Das Problem ist nicht primär „keine Klasse erkannt“, sondern die Auswahl von `x` in ambigen Fällen.

## Dataset Balance Hinweis

`x` (crossed_out) und `f` sind im Trainings-CSV klar unterrepräsentiert:
- pro A-Z Klasse: ca. 2400 Samples
- `f`: 240 Samples
- `x`: 250 Samples

Das kann zu Instabilität bei diesen Randklassen beitragen.

## Decision

Stand jetzt:
- Keine künstliche Aufblähung der `x`-Klasse.
- Ergebnisse und Analyse werden dokumentiert.
- Optionaler nächster Hebel bleibt die Inferenzregel (`prefer_non_x`), falls für Publikationslauf gewünscht.

## Standardisierter Ergebnis-Workflow (ab jetzt)

Ziel: Alle Modelle haben die gleiche Struktur unter `Results/<Model-Name>/`:
- pro Datensatz ein Unterordner `1..61`
- identischer `source_file` je Modell
- eine `_summary.txt` pro Modellordner

### 1) Gewuenschte Modellmenge

Haupttabelle (61/61, beide Modi `-plain` + `-weak`):
- `LLM OpenRouter Gemini-3.1 Flash-Lite`
- `LLM OpenRouter OpenAI GPT-5.2`
- `LLM OpenRouter Qwen3-VL-30B`
- `LLM OpenRouter xAI Grok-4.3`
- `YOLOv5 Medium-plain`
- `YOLO26 Medium-plain`

Nur Premium-Subset (Datasets 1..10), beide Modi, Suffix `-subset_1_10`:
- `LLM OpenRouter Anthropic Claude Opus 4.7`
- `LLM OpenRouter OpenAI GPT-5.5`

Namenskonvention (vereinheitlicht):
- Modus-Suffix: `-plain` (kein Hint) oder `-weak` (Referenz-Hint).
- Laeufe, die nur die Teilmenge `1..10` abdecken, erhalten zusaetzlich `-subset_1_10` (Begriff `short` entfaellt).

### 2) Einheitliche Ablagestruktur

Beispiel:
- `Results/<Model-Name>/_summary.txt`
- `Results/<Model-Name>/1/<source_file>`
- `Results/<Model-Name>/2/<source_file>`
- ...
- `Results/<Model-Name>/61/<source_file>`

### 3) `source_file` pro Modell festlegen (aktuell)

Aktuelle Zuordnung:
- Gemini-3.1 Flash-Lite plain: `recognition_llm_openrouter_gemini31_flashlite.txt`
- Gemini-3.1 Flash-Lite weak: `recognition_llm_openrouter_gemini31_flashlite_weak_hint_ref.txt`
- OpenAI GPT-5.2 plain: `recognition_llm_openrouter_openai_gpt52_plain.txt`
- OpenAI GPT-5.2 weak: `recognition_llm_openrouter_openai_gpt52_weak_hint_ref.txt`
- OpenAI GPT-5.5 plain: `recognition_llm_openrouter_openai_gpt55_plain.txt`
- OpenAI GPT-5.5 weak: `recognition_llm_openrouter_openai_gpt55_weak_hint_ref.txt`
- Anthropic Claude Opus 4.7 plain: `recognition_llm_openrouter_claude_opus47_plain.txt`
- Anthropic Claude Opus 4.7 weak: `recognition_llm_openrouter_claude_opus47_weak_hint_ref.txt`
- Qwen3-VL-30B plain: `recognition_llm_qwen3vl30b_plain.txt`
- Qwen3-VL-30B weak: `recognition_llm_qwen3vl30b_weak_hint_ref.txt`
- xAI Grok-4.3 plain: `recognition_llm_grok43_plain.txt`
- xAI Grok-4.3 weak: `recognition_llm_grok43_weak_hint_ref.txt`
- YOLOv5 Medium: `recognition_yolov5_medium.txt`
- YOLO26 Medium: `recognition_yolo26_medium.txt`

Hinweis:
- Der Dateiname (`source_file`) bleibt stabil (enthaelt weiterhin `weak_hint_ref`); nur der Modell-/Ordnername wird auf `-plain`/`-weak` (+ optional `-subset_1_10`) vereinheitlicht.

Wichtig:
- `source_file` in `_summary.txt` muss exakt mit Dateiname in den Unterordnern `1..61` uebereinstimmen.

### 4) Ergebnisse in den Modellordner kopieren / Summary erzeugen

Empfohlen (nach Inferenzlauf) ueber Helper-Skript:
```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/build_model_results_from_dataset.py \
  --dataset-root data/dataset \
  --results-root Results \
  --model-name "LLM OpenRouter OpenAI GPT-5.2 Plain" \
  --source-file recognition_llm_openrouter_openai_gpt52_plain.txt \
  --provider openrouter \
  --provider-model openai/gpt-5.2
```

Hinweis:
- Das Skript kopiert `data/dataset/<id>/<source_file>` nach `Results/<Model>/<id>/...`.
- Es erzeugt anschliessend automatisch `Results/<Model>/_summary.txt` inkl. TP/FN/FP/TN.

### 5) `_summary.txt` pflicht

Pflichtfelder:
- `model`
- `source_file`
- `datasets_expected`
- `datasets_copied`
- `pages_processed`
- `letters_total`
- `letters_correct`
- `letters_wrong`
- `accuracy`
- `accuracy_percent`
- `extra_pred_tokens`
- `created_at`

Empfohlen zusaetzlich (Publikation):
- `tp`, `fn`, `fp`, `tn`
- `recall`, `fnr`, `precision`, `specificity`

### 6) Globale Vergleichsdateien neu erzeugen

Nach jedem neuen/aktualisierten Modell:
```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/build_classification_metrics_all_results.py
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/build_publication_comparison.py
```

Outputs:
- `Results/classification_metrics_all_results.csv`
- `Results/classification_metrics_all_results.md`
- `Results/publication_comparison/model_overall.csv`
- `Results/publication_comparison/model_per_page.csv`
- `Results/publication_comparison/model_error_types.csv`
- `Results/publication_comparison/model_latency_dataset.csv`
- `Results/publication_comparison/pairwise_mcnemar.csv`

### 7) Abnahme-Checkliste

- Gibt es fuer das Modell `Results/<Model-Name>/_summary.txt`?
- Sind `Results/<Model-Name>/1..61/<source_file>` vollstaendig vorhanden?
- Stimmt `datasets_copied: 61`?
- Ist das Modell in `Results/classification_metrics_all_results.md` sichtbar?

## Latenz-Methodik (Publikation)

Da OpenRouter nicht grok-/modellnativ ist, werden keine request-genauen API-Latenzen als Primarmetrik genutzt.

Fuer Vergleichszwecke verwenden wir eine reproduzierbare Approximation:
- Pro Modell und Datensatz wird `end_ts` aus dem Dateizeitstempel von `Results/<Model>/<dataset_id>/<source_file>` gelesen.
- `start_ts` eines Datensatzes ist `end_ts` des vorherigen Datensatzes (sequenzieller Lauf als Annahme).
- `duration_seconds = end_ts - start_ts`.

Wichtig:
- Diese Metrik ist nur eine **Datensatz-Dauer-Approximation**.
- Sie ist nicht identisch mit echter API-Request-Latenz pro Seite.

## Ablation-Plan (ohne Weak Hint)

Fuer die Publikation werden zusaetzlich diese Plain-Laeufe (`--expected-mode none`) erhoben:
- `LLM OpenRouter OpenAI GPT-5.2 Plain`
- `LLM OpenRouter Qwen3-VL-30B Plain`
- `LLM OpenRouter xAI Grok-4.3 Plain`

Start (resume-faehig, ueberspringt vorhandene Datensatz-Dateien):
```bash
/Users/wiggel/Python/eKlausur2/run_plain_ablation_openrouter.sh
```

Nur Qwen + Grok (wenn GPT-5.2 bereits separat laeuft):
```bash
RUN_GPT52=0 /Users/wiggel/Python/eKlausur2/run_plain_ablation_openrouter.sh
```

Wichtig:
- Keine zusaetzlichen Prompt-/Request-Felder fuer Punkte 2-4 noetig.
- Weitere Kennzahlen werden aus `recognition_*.txt` + `studSolution.txt` + `result.txt` abgeleitet.
