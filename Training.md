# Training

Dieses Dokument beschreibt den Trainingsprozess fuer YOLOv5 und YOLO26.

## Ziel

- Deterministische Trainingslaeufe (Seed-basiert)
- Reproduzierbare Vergleiche zwischen Modellfamilien/Groessen
- Eindeutige Modellbenennung fuer Publikation

## Daten

- Trainingsdaten: `/Users/wiggel/Python/eKlausurData/YoloMultiClassGenerated`
- Data-Config: `/Users/wiggel/Python/eKlausur2/dataset_hg_multiclass_meta.yaml`

## YOLOv5

YOLOv5-Stack:
- `/Users/wiggel/Python/py_yolo/yolov5`

Trainingsskript (in diesem Repo):
- `/Users/wiggel/Python/eKlausur2/train_model_v5.py`

Beispiel (kurzer Testlauf):

```bash
/Users/wiggel/Python/eKlausur2/.venv/bin/python /Users/wiggel/Python/eKlausur2/train_model_v5.py \
  --dataset-dir /Users/wiggel/Python/eKlausurData/YoloMultiClassGenerated \
  --image-ext .png \
  --split-percentage 90 \
  --seed 42 \
  --yolo-root /Users/wiggel/Python/py_yolo/yolov5 \
  --data-config /Users/wiggel/Python/py_yolo/yolov5/dataset_hg_multiclass.yaml \
  --weights yolov5m.pt \
  --hyp hyp_hg_table.yaml \
  --imgsz 640 \
  --epochs 3 \
  --batch -1
```

Bekannte lokale Runs:
- `/Users/wiggel/Python/py_yolo/yolov5/runs/train/good_working_exp17/weights/best.pt`
- `/Users/wiggel/Python/py_yolo/yolov5/runs/train/good_working_seed/weights/best.pt`

## YOLO26

Trainingsskript:
- `/Users/wiggel/Python/eKlausur2/train_model_v26.py`

Beispiel (kurzer Testlauf):

```bash
/Users/wiggel/Python/eKlausur2/.venv312/bin/python /Users/wiggel/Python/eKlausur2/train_model_v26.py \
  --dataset-dir /Users/wiggel/Python/eKlausurData/YoloMultiClassGenerated \
  --image-ext .png \
  --split-percentage 90 \
  --seed 42 \
  --data-config /Users/wiggel/Python/eKlausur2/dataset_hg_multiclass_meta.yaml \
  --model yolo26m.pt \
  --imgsz 640 \
  --epochs 3 \
  --batch 16 \
  --device 0
```

## Lokaler Modellordner (zentral)

Alle Arbeitsmodelle zentral in:
- `/Users/wiggel/Python/eKlausur2/models_local/`

Wichtig:
- `models_local/` ist lokal und wird nicht gepusht.
- Verteilbare Modelle kommen in GitHub Releases (Assets).

## Release-Konvention (GitHub Assets)

Release-Seite:
- `https://github.com/grabow/eKlausur/releases`

Namenskonvention:
- Release-Titel: `Model <FAMILY> <SIZE> (<YYYY-MM-DD>)`
- Asset-Datei: `<family><size>_<YYYYMMDD>_best.pt`

Beispiele:
- `Model YOLOv5 Medium (2026-05-06)`
- `yolov5m_20260506_best.pt`
- `yolo26m_20260507_best.pt`
- `yolo26l_20260507_best.pt`

## Windows 3090 Schnellstart

Im Repo-Root liegen Runner-Skripte fuer Windows:
- `/Users/wiggel/Python/eKlausur2/run_training_3090_v5.bat`
- `/Users/wiggel/Python/eKlausur2/run_training_3090_v26.bat`

Hinweis:
- Vor dem Start die Pfade im jeweiligen `.bat` anpassen (`DATASET_DIR`, bei v5 auch `YOLO_ROOT`).
