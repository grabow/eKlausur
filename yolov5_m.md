# YOLOv5 Seed-Modell: Inferenzlauf `/1` bis `/61`

## Ziel
Dokumentation eines reproduzierbaren YOLOv5-Inferenzlaufs mit dem deterministisch trainierten Seed-Modell fuer spaetere Publikation.

## Modell
- Verwendetes Modell: `/Users/wiggel/Python/py_yolo/yolov5/runs/train/good_working_seed/weights/best.pt`
- Alias im Projekt: `model_seed`

## Datensatzbereich
- Evaluierte Datasets: `data/dataset/1` bis `data/dataset/61`
- Verarbeitete Seiten: `742`

## Inferenzaufruf
Interpreter:
- `/Users/wiggel/Python/eKlausur2/.venv/bin/python`

Aufruf:
```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/run_yolov5_recognition.py \
  --dataset-root /Users/wiggel/Python/eKlausur2/.tmp_eval_1_61 \
  --yolo-root /Users/wiggel/Python/py_yolo/yolov5 \
  --model-path /Users/wiggel/Python/py_yolo/yolov5/runs/train/good_working_seed/weights/best.pt \
  --output-name recognition_seed_det.txt
```

Hinweise:
- Pro Dataset wurde `recognition_seed_det.txt` erzeugt.
- Es gab keine Seiten-Fallbacks (`?`) durch Laufzeitfehler.

## Bewertungsregel (Token-basiert)
Pro Zeile wird gegen `studSolution.txt` tokenweise bewertet.

Wichtig fuer die Publikation:
- Falls die Vorhersage laenger als die Soll-Zeile ist, werden nur die ersten `x` Tokens betrachtet, wobei `x = Anzahl Soll-Tokens`.
- Zusatztokens hinter Position `x` werden nicht in die Punktzahl aufgenommen (nur separat berichtet).

## Ergebnisse
- Buchstaben insgesamt (Soll-Tokens): `3141`
- Korrekt erkannt: `2856`
- Falsch erkannt: `285`
- Accuracy: `2856 / 3141 = 0.909265` (`90.93%`)
- Zusaetzliche vorhergesagte Tokens (hinter Soll-Laenge): `24`

## Ausgabeorte
- Vorhersagedateien je Dataset:
  - `/Users/wiggel/Python/eKlausur2/data/dataset/<id>/recognition_seed_det.txt`
- Referenz je Dataset:
  - `/Users/wiggel/Python/eKlausur2/data/dataset/<id>/studSolution.txt`
