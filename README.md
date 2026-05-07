# eKlausur2

Eigenstaendiger, Python-basierter Workflow fuer Training, Inferenz und Evaluation zur automatischen Klausurkorrektur.
Methodische Referenz:
- `/Users/wiggel/IntelliJIDEA/eKlausur`

## Projektfokus

- Datensammlung/-aufbereitung fuer Evaluation
- Modellentraining (YOLOv5, YOLO26)
- Reproduzierbare Inferenz und tokenbasierte Auswertung

## Laufzeit-Hinweise

- Inferenz-Interpreter:
  - `/Users/wiggel/Python/eKlausur2/.venv/bin/python`
- V26-Training typischerweise mit:
  - `/Users/wiggel/Python/eKlausur2/.venv312/bin/python`

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
