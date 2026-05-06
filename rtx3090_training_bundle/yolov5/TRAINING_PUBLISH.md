# Reproducible Training (eKlausur YOLOv5)

This folder contains a reproducible training entrypoint for the eKlausur YOLOv5 model:

- Script: `train_model.py`
- Split strategy: deterministic 90/10 train/validation split with fixed seed
- Model family: YOLOv5 (`train.py`)

## What was changed for reproducibility

- Fixed random seed support (`--seed`, default `42`)
- Deterministic split creation using seeded shuffle
- Split manifest written to:
  - `split_manifest_seed.txt` in dataset root
- Global seeds set for:
  - Python `random`
  - NumPy (if installed)
  - PyTorch (if installed)

## Dataset expectations

The dataset directory must contain paired files:

- `<sample>.txt` (YOLO labels)
- `<sample>.png` (or custom extension via `--image-ext`)

The script creates/overwrites:

- `images/training`, `images/validation`
- `labels/training`, `labels/validation`

inside the dataset directory.

## Recommended run

```bash
cd /Users/wiggel/Python/py_yolo/yolov5

# set dataset location (recommended for portability)
export YOLO_HG_DATASET_DIR=/path/to/YoloMultiClassGenerated

# verify split only (no training)
python3 train_model.py --dry-run --seed 42 --split-percentage 90 --image-ext .png

# full training
python3 train_model.py \
  --seed 42 \
  --split-percentage 90 \
  --image-ext .png \
  --data-config dataset_hg_multiclass.yaml \
  --imgsz 640 \
  --weights yolov5s.pt \
  --hyp hyp_hg_table.yaml \
  --epochs 20 \
  --batch -1
```

## Notes for publication

- Report the exact `--seed`, `--split-percentage`, and dataset path.
- Archive `split_manifest_seed.txt` alongside results for full replayability.
- Report the produced YOLO run directory (`runs/train/...`) and selected checkpoint (`weights/best.pt`).
