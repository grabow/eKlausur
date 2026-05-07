# Inference

Dieses Dokument beschreibt den standardisierten Inferenzlauf und die Ergebnisbewertung.

## Interpreter

Fuer Inferenzlaeufe verwenden:
- `/Users/wiggel/Python/eKlausur2/.venv/bin/python`

## API-Keys (autark in eKlausur2)

- Lokale Keys liegen in:
  - `/Users/wiggel/Python/eKlausur2/.env`
- `run_llm_recognition.py` laedt diese Datei automatisch (Parameter `--env-file`, Default `.env`).

## Referenzprojekt

Die fachliche Referenz bleibt:
- `/Users/wiggel/IntelliJIDEA/eKlausur`

## Skripte

- `/Users/wiggel/Python/eKlausur2/run_yolov5_recognition.py`
- `/Users/wiggel/Python/eKlausur2/run_yolo26_recognition.py`
- `/Users/wiggel/Python/eKlausur2/run_llm_recognition.py`

Wichtige Parameter:
- `--dataset-root` (z. B. `data/dataset`)
- `--model-path` (direkte Modellwahl aus lokalem Modellordner)
- `--output-name` (Dateiname pro Dataset)

YOLOv5-spezifisch:
- `run_yolov5_recognition.py` nutzt den eKlausur/py_yolo-Stack (`get_results_yolo_all`, `find_boxes`)
- zusaetzlich: `--yolo-root`, `--cleanup-box-png`

YOLO26-spezifisch:
- `run_yolo26_recognition.py` nutzt Ultralytics direkt
- zusaetzlich: `--conf`, `--iou`, `--imgsz`, `--max-det`, `--line-y-ratio`, `--device`

LLM-spezifisch:
- `run_llm_recognition.py` nutzt denselben Foundation-Model-Stack wie eKlausur (`recognizer.copy_blurr_resize` + `recognizer.recognize`)
- Provider waehlen: `--provider openai|gemini|openrouter|academiccloud|ollama`
- Konkretes Modell waehlen: `--provider-model <modellname>`
- Dataset-Auswahl: `--dataset-id 1` (mehrfach moeglich)
- zusaetzlich: `--prompt-index`, `--expected-mode`, `--raw-json-dir`, `--log-file`

Beispiele:
```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/run_llm_recognition.py \
  --dataset-root /Users/wiggel/Python/eKlausur2/data/dataset \
  --dataset-id 1 \
  --provider openai \
  --provider-model gpt-5.2-2025-12-11 \
  --output-name recognition_llm_openai.txt
```

```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/run_llm_recognition.py \
  --dataset-root /Users/wiggel/Python/eKlausur2/data/dataset \
  --dataset-id 1 \
  --provider gemini \
  --provider-model gemini-2.5-flash \
  --output-name recognition_llm_gemini.txt
```

## Modellablage fuer Inferenz

Lokaler Sammelordner (nicht versioniert):
- `/Users/wiggel/Python/eKlausur2/models_local/`

Beispielmodelle:
- `yolov5m_seed.pt`
- `yolov5m_good_working_exp17.pt`

Hinweis:
- Modell muss nicht nach `py_yolo` kopiert werden, wenn `--model-path` gesetzt ist.
- Fuer YOLO26 ist ein separates Skript erforderlich; `run_yolov5_recognition.py` ist an den YOLOv5/eKlausur-Stack gebunden.

## Ergebnisablage

Ergebnisse werden pro Modell unter `Results/<Model-Name>/` abgelegt.

Beispiel:
- `Results/YOLOv5 Medium/1/recognition_seed_det.txt`
- `Results/YOLOv5 Medium/61/recognition_seed_det.txt`
- `Results/YOLOv5 Medium/_summary.txt`

## Bewertungsregel (tokenbasiert)

Vergleich gegen `studSolution.txt`:
- Bewertet werden pro Zeile nur die ersten `x` Vorhersagetokens, mit `x = Anzahl Soll-Tokens`.
- Zusatztokens werden separat als `extra_pred_tokens` gezaehlt.
- `letters_wrong = letters_total - letters_correct`.

## Pflichtfelder in `_summary.txt`

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

## Standard-Kennzahlen

- `letters_total`: Gesamtzahl Soll-Tokens
- `letters_correct`: korrekt erkannte Tokens
- `letters_wrong`: falsch erkannte Tokens
- `accuracy`: `letters_correct / letters_total`
- `accuracy_percent`: `accuracy * 100`
- `pages_processed`: Anzahl bewerteter Seiten
- `extra_pred_tokens`: vorhergesagte Tokens ohne GT-Pendant
