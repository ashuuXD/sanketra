"""Measured latency/throughput/size for a model on THIS machine (RoadSight RQ4).
  python -m evaluation.benchmark_latency --models yolo11n.pt yolo11s.pt --imgsz 320 416 640
"""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np

from vision.paths import HOME

RESULTS = HOME / "experiments" / "results"


def bench(model_path: str, imgsz: int, device: str, runs: int = 50, warmup: int = 10) -> dict:
    from ultralytics import YOLO
    m = YOLO(model_path)
    img = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
    for _ in range(warmup):
        m.predict(img, imgsz=imgsz, device=device, verbose=False)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); m.predict(img, imgsz=imgsz, device=device, verbose=False); t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    p = Path(model_path)
    return {"model": p.name, "imgsz": imgsz, "device": device, "runs": runs, "median_ms": round(float(np.median(t)), 1),
            "p95_ms": round(float(np.percentile(t, 95)), 1), "fps_from_median": round(1000 / float(np.median(t)), 1),
            "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt"]); ap.add_argument("--imgsz", type=int, nargs="+", default=[320, 416, 640])
    ap.add_argument("--device", default="auto"); ap.add_argument("--runs", type=int, default=50); ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    import torch
    dev = ("cuda:0" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    rows = [bench(m, s, dev, a.runs) for m in a.models for s in a.imgsz]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"machine": platform.platform(), "device": dev, "rows": rows}
    (RESULTS / f"latency_{a.tag}.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(r)
