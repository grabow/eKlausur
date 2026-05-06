# YOLOv5 Training Bundle fuer Windows (RTX 3090)

Dieses Paket enthaelt:
- `yolov5/` (Training-Code)
- `data/YoloMultiClassGenerated/` (komplette Trainingsdaten)

## 1) Voraussetzungen
- Windows 10/11
- NVIDIA Treiber fuer RTX 3090 (aktuell)
- Python 3.12.x (64-bit)
- Genug Plattenplatz (entpackt mehrere GB)

## 2) Setup in PowerShell
Im entpackten Ordner ausfuehren:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# CUDA-Wheels fuer NVIDIA (RTX 3090)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Pin wie gewuenscht
pip install "numpy>=1.26,<1.27"

# YOLO-Abhaengigkeiten
pip install -r .\yolov5\requirements.txt
```

## 3) GPU-Check
```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 4) Training starten (3 Epochen, ohne --noval)
```powershell
python .\yolov5\train_model.py `
  --dataset-dir .\data\YoloMultiClassGenerated `
  --image-ext .png `
  --split-percentage 90 `
  --seed 42 `
  --data-config .\yolov5\dataset_hg_multiclass_windows.yaml `
  --imgsz 640 `
  --weights .\yolov5\yolov5s.pt `
  --hyp .\yolov5\hyp_hg_table.yaml `
  --epochs 3 `
  --batch 16 `
  --device 0
```

## 5) Ergebnisdateien
- Gewichte: `yolov5\runs\train\exp*\weights\best.pt`
- Reproduzierbarer Split: `data\YoloMultiClassGenerated\split_manifest_seed.txt`

## Hinweise
- Fuer erneute Runs mit gleichem Split wieder denselben `--seed` nutzen.
- Bei VRAM-Problemen `--batch 8` oder `--imgsz 512` testen.
