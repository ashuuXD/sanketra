"""Milestone 1: prove the model produces a valid detection. Run: python scripts/smoke_test.py"""
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

root = Path(__file__).resolve().parent.parent
img = root / "tests" / "assets" / "street.jpg"
w = root / "models" / "yolo11n.pt"
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
model = YOLO(str(w) if w.exists() else "yolo11n.pt")
r = model(str(img), conf=0.25, verbose=False)[0]
names = [model.names[int(c)] for c in r.boxes.cls]
print("detections:", names)
r.save(filename=str(root / "smoke_out.jpg")); print("saved smoke_out.jpg")
sys.exit(0 if names else 1)
