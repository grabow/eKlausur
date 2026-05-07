@echo off
setlocal

REM Adjust this path for your Windows machine
set DATASET_DIR=D:\eKlausurData\YoloMultiClassGenerated

py -3.12 train_model_v26.py ^
  --dataset-dir "%DATASET_DIR%" ^
  --image-ext .png ^
  --split-percentage 90 ^
  --seed 42 ^
  --data-config dataset_hg_multiclass_meta.yaml ^
  --model yolo26m.pt ^
  --imgsz 640 ^
  --epochs 20 ^
  --batch 16 ^
  --device 0

endlocal
