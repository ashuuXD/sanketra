"""Calibration study (RoadSight RQ2): reliability diagram, ECE, temperature scaling.
T is fitted on the VAL split and evaluated on the TEST split (no leakage).
  python -m evaluation.calibration_eval --model models/best.pt --data datasets/prepared/gtsdb/data.yaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from vision.calibration import apply_temperature, ece, fit_temperature, match_predictions

from vision.paths import HOME

RESULTS = HOME / "experiments" / "results"


def read_gt(lbl: Path, w: int, h: int):
    out = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            c, x, y, bw, bh = line.split()[:5]
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            out.append((int(c), (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)))
    return out


def collect(model, data_yaml: Path, split: str, imgsz: int, device):
    cfg = yaml.safe_load(data_yaml.read_text()); root = Path(cfg["path"])
    img_dir = root / cfg[split]; lbl_dir = root / cfg[split].replace("images", "labels")
    confs, correct = [], []
    for p in sorted(img_dir.iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, conf=0.05, imgsz=imgsz, device=device, verbose=False)[0]
        preds = [(int(k), float(c), b) for k, c, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        c_, ok_, _ = match_predictions(preds, read_gt(lbl_dir / f"{p.stem}.txt", w, h))
        confs += c_.tolist(); correct += ok_.tolist()
    return np.array(confs), np.array(correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--device", default=None); ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.model)
    cv_, yv = collect(m, a.data, "val", a.imgsz, a.device)
    ct, yt = collect(m, a.data, "test", a.imgsz, a.device)
    T = fit_temperature(cv_, yv)
    e0, b0 = ece(ct, yt); e1, b1 = ece(apply_temperature(ct, T), yt)
    res = {"temperature_fitted_on_val": T, "n_val": int(len(cv_)), "n_test": int(len(ct)),
           "ece_before": e0, "ece_after": e1, "bins_before": b0, "bins_after": b1}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"calibration_{a.tag}.json").write_text(json.dumps(res, indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "--", c="gray")
    for bins, lab in ((b0, f"raw ECE={e0:.3f}"), (b1, f"T={T:.2f} ECE={e1:.3f}")):
        pts = [(b["conf"], b["acc"]) for b in bins if b["n"]]
        ax.plot(*zip(*pts), marker="o", label=lab)
    ax.set_xlabel("confidence"); ax.set_ylabel("precision in bin"); ax.legend(); ax.set_title("Reliability (test split)")
    fig.tight_layout(); fig.savefig(RESULTS / f"calibration_{a.tag}.png", dpi=130)
    print(json.dumps({k: res[k] for k in ("temperature_fitted_on_val", "ece_before", "ece_after")}, indent=2))


if __name__ == "__main__":
    main()
