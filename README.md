# eKlausur2

Dieses Projekt ist ein eigenständiger, Python-basierter Evaluations-Workflow zur automatischen Klausurkorrektur.
Die fachliche und methodische Referenz ist das bestehende Projekt **eKlausur**.

## Ziel

Wir wollen die Erkennungsleistung verschiedener Ansätze vergleichbar evaluieren, insbesondere:
- YOLO-basierter Ansatz
- Foundation-Model-Ansätze (z. B. Gemini, OpenAI, weitere)
- Optional ein redundanter Zwei-Stufen-Ansatz (zweites Modell zur Validierung)

## Vorgehen

1. **Daten sammeln**
   - Einwilligungslisten (CSV/XLSX) auswerten.
   - Relevante Matrikelordner aus mehreren Quellverzeichnissen in ein gemeinsames Dataset-Verzeichnis kopieren.
   - Sensible Daten nur gemäß Freigabe und anonymisiert bereitstellen.
   - Lokales Arbeitsverzeichnis dafür: `./data/` (wird nicht eingecheckt, siehe `.gitignore`)
   - Anleitung/Beispiel: `data/README.md`

2. **Python-Skripte schreiben**
   - Skript für Datensammlung/Filterung (`collect_exam_data.py`).
   - Skript(e) für Erkennungsläufe (YOLO + Foundation Models) auf denselben Daten.
   - Optional Skript für Zwei-Stufen-Redundanz (Erkennung + Gegenprüfung durch zweites Modell).

3. **Auswerten**
   - Einheitliche Metriken berechnen (z. B. tokenweise Genauigkeit, Fehlerraten pro Klausurtyp).
   - Ergebnisse pro Modell und pro Klausurtyp ausgeben.
   - Vergleichstabellen/CSVs für Paper und Reproduzierbarkeit erzeugen.

## Referenz: eKlausur

Die technische Umsetzung hier ist bewusst unabhängig von der eKlausur-GUI.
Das methodische Vorgehen orientiert sich aber an eKlausur:
- Lokaler Referenzpfad: `/Users/wiggel/IntelliJIDEA/eKlausur`
- gleiche inhaltliche Aufgabenstellung,
- vergleichbare Eingabe-/Ausgabeformate,
- reproduzierbare Evaluationslogik.

## Laufzeit-Hinweis

Für Läufe bitte diesen Interpreter nutzen:
`/Users/wiggel/Python/eKlausur2/.venv/bin/python`

## Modell-Download & Smoke-Test (macOS)

- Download nach `/Users/wiggel/Python/eKlausur2/tmp_models/best.pt`
- Test mit `/Users/wiggel/Python/eKlausur2/.venv/bin/python` (Torch 2.11.0) über `run_yolov5_recognition.py`
- Hinweis: Anfangs gab’s das bekannte Windows→macOS-Problem (WindowsPath im Checkpoint). `run_yolov5_recognition.py` wurde so gepatcht, dass `torch.load` bei WindowsPath automatisch retryt (WindowsPath→PosixPath).

## YOLOv26 Training (deterministisch)

Zusätzlich zur YOLOv5-Pipeline gibt es im Repo ein zweites Trainingsskript:

- `/Users/wiggel/Python/eKlausur2/train_model_v26.py`

Eigenschaften:
- gleicher deterministischer Split-Workflow (Seed, `split_manifest_seed.txt`)
- automatische Runtime-Data-YAML auf Basis des aktuellen Splits
- Training über Ultralytics-API mit `--model yolo26m.pt` oder `--model yolo26l.pt`

Beispiel (macOS/Local):

```bash
/Users/wiggel/Python/eKlausur2/.venv312/bin/python /Users/wiggel/Python/eKlausur2/train_model_v26.py \
  --dataset-dir /Users/wiggel/Python/eKlausurData/YoloMultiClassGenerated \
  --image-ext .png \
  --split-percentage 90 \
  --seed 42 \
  --data-config /Users/wiggel/Python/eKlausur2/dataset_hg_multiclass_meta.yaml \
  --model yolo26m.pt \
  --imgsz 640 \
  --epochs 20 \
  --batch 16 \
  --device 0
```

Hinweis:
- fuer rein reproduzierbare Publikationslaeufe immer `--seed` dokumentieren
- fuer Vergleichslaufe `yolo26m.pt` und `yolo26l.pt` mit identischem Split/Seed ausfuehren

## GitHub Zugriff (SSH + GH CLI)

SSH fuer GitHub ist eingerichtet.
Test:

```powershell
ssh -T git@github.com
```

Erwartete Rueckmeldung:
`Hi grabow! You've successfully authenticated, but GitHub does not provide shell access.`

GitHub CLI ist installiert unter:
`C:\Program Files\GitHub CLI\gh.exe`

Falls `gh` im aktuellen Terminal nicht gefunden wird, ein neues Terminal oeffnen
oder direkt aufrufen:

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" --version
```

## Modell-Releases (GitHub Assets)

Trainierte Modelle werden fuer die Nachnutzung (z. B. 4090-Training/Inference) als **GitHub Release Assets** im Repository `grabow/eKlausur` abgelegt.

Release-Seite:
- `https://github.com/grabow/eKlausur/releases`

### Namenskonvention

Release-Titel:
- `Model <FAMILY> <SIZE> (<YYYY-MM-DD>)`
- Beispiel: `Model YOLOv5 Medium (2026-05-06)`

Asset-Dateiname:
- `<family><size>_<YYYYMMDD>_best.pt`
- Beispiele:
  - `yolov5m_20260506_best.pt`
  - `yolo26m_20260507_best.pt`
  - `yolo26l_20260507_best.pt`

Hinweis:
- Keine experimentellen Run-IDs (`exp*`) im finalen Asset-Namen.
- So bleibt die Zuordnung fuer Publikation und Reproduktion stabil und eindeutig.

### Upload-Workflow (GitHub CLI)

1. Release anlegen (oder bestehenden Tag verwenden):

```bash
gh release create <tag> \
  --repo grabow/eKlausur \
  --title "Model YOLOv5 Medium (2026-05-06)" \
  --notes "Trained checkpoint for publication."
```

2. Modell lokal auf den finalen Namen bringen und als Asset hochladen:

```bash
cp /path/to/best.pt /tmp/yolov5m_20260506_best.pt
gh release upload <tag> /tmp/yolov5m_20260506_best.pt --repo grabow/eKlausur
```

3. Falls ein Asset bereits als `best.pt` existiert, zuerst loeschen und mit finalem Namen neu hochladen:

```bash
gh release delete-asset <tag> best.pt --repo grabow/eKlausur --yes
gh release upload <tag> /tmp/yolov5m_20260506_best.pt --repo grabow/eKlausur
```
