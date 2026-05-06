# YOLOv5 Recognition Pipeline (`run_yolov5_recognition.py`)

## Zweck

`run_yolov5_recognition.py` erzeugt für jedes Dataset im Ordner `data/dataset/<id>/` eine Datei `recognition_yolov5.txt` mit den erkannten Buchstaben pro Seite.

- Input pro Dataset: `page_*.jpg`, `studSolution.txt`
- Output pro Dataset: `recognition_yolov5.txt`
- Ausgabeformat: eine Zeile pro Seite (`page_i.jpg`), Tokens mit Leerzeichen getrennt (z. B. `A B C`)

Die Pipeline ist auf das bestehende eKlausur-Setup abgestimmt und nutzt den vorhandenen YOLOv5-Stack aus:

- `/Users/wiggel/Python/py_yolo/yolov5`


## Implementierungsprinzip

Das Skript nutzt den bestehenden Erkennungsaufruf:

- `from get_results_yolo_all import get_result_list`

und verarbeitet pro Seite:

1. **eKlausur-kompatible Vorverarbeitung** (äquivalent zu `copy_invert_blurr`):
   - Graustufen
   - Gaussian Blur (`k=9` bei <256 Grauwerten, sonst `k=7`)
   - Invertierung (`255 - image`)
2. **YOLOv5-Erkennung** über `get_result_list(...)`
3. **Normalisierung der Tokens** auf `A-Z` bzw. `?`
4. **Schreiben in `recognition_yolov5.txt`**

Zusätzlich wird ein Kompatibilitäts-Patch für Legacy-Checkpoints gesetzt:

- `torch.load(..., weights_only=False)` (falls nicht explizit gesetzt)

Damit lassen sich ältere YOLOv5-Modelle unter neueren PyTorch-Versionen laden.


## Voraussetzungen

Empfohlener Interpreter:

- `/Users/wiggel/Python/eKlausur2/.venv/bin/python`

Erforderliche Bibliotheken (in der venv installiert):

- `torch`
- `opencv-python`
- `pillow`
- `pyyaml`
- YOLOv5-Requirements aus `py_yolo/yolov5/requirements.txt`


## CLI

```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/run_yolov5_recognition.py [OPTIONS]
```

Optionen:

- `--dataset-root` (Default: `data/dataset`)
  - Root mit numerischen Dataset-Ordnern (`1`, `2`, ...)
- `--yolo-root` (Default: `/Users/wiggel/Python/py_yolo/yolov5`)
  - YOLOv5-Funktionspfad (`get_results_yolo_all.py`, `find_boxes.py`, Modellreferenzen)
- `--output-name` (Default: `recognition_yolov5.txt`)
  - Ziel-Dateiname pro Dataset
- `--fail-token` (Default: `?`)
  - Fallback-Token bei seitenweisem Fehler oder leerer Erkennung
- `--cleanup-box-png / --no-cleanup-box-png` (Default: Cleanup aktiv)
  - Löscht temporäre YOLO-Crop-Dateien `*_box_*.png` nach Verarbeitung


## Beispiel

Voller Lauf über alle Datasets:

```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/run_yolov5_recognition.py
```

Lauf nur für ein Dataset (z. B. `2`):

```bash
tmp_root=$(mktemp -d)
ln -s /Users/wiggel/Python/eKlausur2/data/dataset/2 "$tmp_root/2"
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/run_yolov5_recognition.py --dataset-root "$tmp_root"
rm -rf "$tmp_root"
```


## Ausgabevertrag

Für jedes `data/dataset/<id>/` mit `studSolution.txt` gilt:

- `recognition_yolov5.txt` wird erzeugt/überschrieben
- Anzahl Zeilen = Anzahl `page_*.jpg`
- Zeile `i` entspricht `page_i.jpg`
- Erlaubte Tokens: `A-Z` und `?`


## Qualität und bekannte Grenzen

- Die Pipeline ist robust für Batch-Läufe (seitenweise Fehler führen nicht zum Komplettabbruch).
- Typische Restfehler sind visuell ähnliche Buchstaben, insbesondere `I` vs `J`.
- Ergebnisqualität hängt direkt vom in `find_boxes.py` konfigurierten Modell (`best.pt`) und dessen Trainingsstand ab.


## Reproduzierbarkeit / Hinweise

- Das Skript ändert **nicht** den Code im eKlausur-Projekt (`/Users/wiggel/IntelliJIDEA/eKlausur`).
- Temporäre Preprocessing-Dateien werden in einem `TemporaryDirectory` erzeugt und automatisch entfernt.
- YOLO-seitig erzeugte `*_box_*.png` werden standardmäßig nach dem Lauf bereinigt.
