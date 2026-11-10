"""Measure, on THIS machine, what actually makes inference fast or slow. Run it with the server STOPPED (Ctrl+C):
    python scripts\\bench_gpu.py
For each setting it prints: first call (includes any one-time shape setup), then steady-state median and p95 in ms.
Settings compared: rectangular vs square input, FP16 vs FP32 (GPU only), at sizes 640 and 416,
plus the cost of drawing + JPEG-encoding a frame. Send the printed table to whoever is tuning the app."""
import argparse, platform, statistics, sys, time
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(root))
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=str(root / "models" / "yolo11n.pt")); ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {dev} | {torch.cuda.get_device_name(0) if dev != 'cpu' else platform.processor()}")
src = cv2.imread(str(root / "data" / "samples" / "street_bus.jpg"))
frames = {"960x540": cv2.resize(src[:540 * src.shape[1] // 960], (960, 540)) if False else cv2.resize(src[100:100 + int(src.shape[1] * 9 / 16)], (960, 540)),
          "640x480": cv2.resize(src[100:100 + int(src.shape[1] * 3 / 4)], (640, 480))}
fp16 = ({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}) if dev != "cpu" else {}
variants = [("rect  fp16", dict(rect=True, **fp16)), ("square fp16", dict(rect=False, **fp16))] if fp16 else []
variants += [("rect  fp32", dict(rect=True)), ("square fp32", dict(rect=False))]
print(f"{'setting':12} {'imgsz':>5} {'frame':>8} {'first ms':>9} {'median':>8} {'p95':>7}")
for imgsz in (640, 416):
    for name, kw in variants:
        m = YOLO(a.model)
        for fname, fr in frames.items():
            t = []
            for _ in range(a.n + 1):
                t0 = time.perf_counter(); m.predict(fr, imgsz=imgsz, device=dev, verbose=False, **kw); t.append((time.perf_counter() - t0) * 1000)
            st = sorted(t[1:])
            print(f"{name:12} {imgsz:5d} {fname:>8} {t[0]:9.0f} {statistics.median(st):8.1f} {st[int(len(st) * .95) - 1]:7.1f}")
# drawing + encoding cost of one frame
from vision.draw import draw_detections
d = [{"box": [100, 100, 400, 400], "uncertain": False, "category": "unknown", "label_en": "Person", "conf": .9, "track_id": 1}] * 5
t = []
for _ in range(50):
    t0 = time.perf_counter(); ok, buf = cv2.imencode(".jpg", draw_detections(frames["960x540"], d), [cv2.IMWRITE_JPEG_QUALITY, 70]); t.append((time.perf_counter() - t0) * 1000)
print(f"draw + JPEG encode of one 960x540 frame: median {statistics.median(t):.1f} ms")
