@echo off
setlocal

if not exist .venv (
  py -3.12 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install "numpy>=1.26,<1.27"
pip install -r .\yolov5\requirements.txt

python .\yolov5\train_model.py ^
  --dataset-dir .\data\YoloMultiClassGenerated ^
  --image-ext .png ^
  --split-percentage 90 ^
  --seed 42 ^
  --data-config .\yolov5\dataset_hg_multiclass_windows.yaml ^
  --imgsz 640 ^
  --weights .\yolov5\yolov5s.pt ^
  --hyp .\yolov5\hyp_hg_table.yaml ^
  --epochs 3 ^
  --batch 16 ^
  --device 0

endlocal
