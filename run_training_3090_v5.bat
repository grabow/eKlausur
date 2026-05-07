@echo off
setlocal

REM Adjust these paths for your Windows machine
set DATASET_DIR=D:\eKlausurData\YoloMultiClassGenerated
set YOLO_ROOT=D:\py_yolo\yolov5

py -3.12 train_model_v5.py ^
  --dataset-dir "%DATASET_DIR%" ^
  --image-ext .png ^
  --split-percentage 90 ^
  --seed 42 ^
  --yolo-root "%YOLO_ROOT%" ^
  --data-config "%YOLO_ROOT%\dataset_hg_multiclass.yaml" ^
  --weights yolov5m.pt ^
  --hyp hyp_hg_table.yaml ^
  --imgsz 640 ^
  --epochs 20 ^
  --batch -1

endlocal
