# eKlausur2

Eigenstaendiger, Python-basierter Workflow fuer Training, Inferenz und Evaluation zur automatischen Klausurkorrektur.
Methodische Referenz:
- `/Users/wiggel/IntelliJIDEA/eKlausur`

## Projektfokus

- Datensammlung/-aufbereitung fuer Evaluation
- Modellentraining (YOLOv5, YOLO26)
- Reproduzierbare Inferenz und tokenbasierte Auswertung (YOLO + Foundation-Modelle)

## Laufzeit-Hinweise

- Inferenz-Interpreter:
  - `/Users/wiggel/Python/eKlausur2/.venv/bin/python`
- V26-Training typischerweise mit:
  - `/Users/wiggel/Python/eKlausur2/.venv312/bin/python`
- API-Keys (nur diese Quelle):
  - `/Users/wiggel/Python/eKlausur2/.env`
  - `run_llm_recognition.py` nutzt ausschliesslich `.env` (kein `credentials.txt`-Fallback)

## Gemini 3.5 Flash - bekannte Falle

- Beobachtung (reproduziert):
  - Bei `provider=gemini` + `model=gemini-3.5-flash` kann dieselbe Seite je nach Bildvariante unterschiedlich reagieren:
    - Originalbild (z. B. `data/dataset/10/page_1.jpg`) antwortet normal.
    - Vorverarbeitetes Bild aus `copy_blurr_resize(...)` kann in API-Calls auf `Read timed out (60s)` laufen.
- Wichtig:
  - Das ist nicht automatisch ein Prompt-/Schema-Fehler.
  - In diesem Fall endet der Request als `None`, wodurch `run_llm_recognition.py` ein `?` schreibt.
- Standard-Vorgehen fuer kuenftige Tests:
  - Bei Gemini 3.5 Flash zuerst mit Originalbild gegenpruefen (ohne Preprocessing).
  - Timeouts separat als Transport/Provider-Thema behandeln, nicht als OCR-Inhaltsfehler.
  - Bei Publikationsvergleichen Protokoll klar kennzeichnen:
    - `plain/common preprocessing` vs. `provider-native robust`.

## Doku

- Inferenzablauf und Ergebnisformat:
  - `/Users/wiggel/Python/eKlausur2/Inference.md`
- Trainingsprozess (YOLOv5/YOLO26), Modellablage und Release-Konvention:
  - `/Users/wiggel/Python/eKlausur2/Training.md`

## Wichtige lokale Ordner

- Daten (nicht versioniert): `./data/`
- Lokale Modelle (nicht versioniert): `./models_local/`
- Ergebnisse je Modell: `./Results/<Model-Name>/`

## Datensammlungsskripte

- `/Users/wiggel/Python/eKlausur2/data_collector_scripts/collect_exam_data.py`

## Windows-Runner (3090)

- `/Users/wiggel/Python/eKlausur2/run_training_3090_v5.bat`
- `/Users/wiggel/Python/eKlausur2/run_training_3090_v26.bat`
