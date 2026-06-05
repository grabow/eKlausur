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

## Gemini 3.5 Flash - Provider-Hinweise (Stand 2026-06-05)

- Aktueller Stand: Der frueher beobachtete `Read timed out (60s)` bei `provider=gemini` +
  `gemini-3.5-flash` auf vorverarbeiteten Bildern ist **aktuell nicht mehr reproduzierbar**.
  Ein voller Google-direct-Lauf (61/61, plain + weak) lief sauber durch (~0.9 % `?`).
  Der Timeout war offenbar ein transientes Transport-/Provider-Problem.
- OpenRouter dagegen (`google/gemini-3.5-flash`): **kein** Timeout, aber gehaeufte
  Seitenausfaelle mit `Antwort ist kein parsebares JSON-Objekt` (Schema-Compliance) ->
  kuenstlich erhoehte `?`-Rate (ca. doppelt so hoch wie Google-direct). Verfaelscht den Vergleich.
- Empfehlung: Gemini 3.5 Flash fuer Publikationslaeufe ueber **Google-direct** (`--provider gemini`)
  fahren, nicht ueber OpenRouter.
- Allgemein bei `None`-Antworten: nicht automatisch als OCR-Inhaltsfehler werten; Transport-/
  Schema-Probleme separat behandeln. Komplette `?`-Zeilen mit lesbarem Ground Truth sind ein
  Indikator fuer Provider-Ausfall (siehe Abbruch-Guard in den Runnern + `check_openrouter_credit.py`).

- Ergebnis (61/61): Gemini 3.5 Flash ist statistisch gleichauf mit 3.1 Flash-Lite
  (weak 98.22 % vs 98.38 %, CIs ueberlappen) -> die neuere Flash-Generation bringt fuer diese
  Aufgabe keinen belastbaren Vorteil.

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
