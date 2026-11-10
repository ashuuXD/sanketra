"""Robustness study: detector accuracy vs controlled corruption severity (RoadSight RQ1).

Reads a YOLO dataset yaml, corrupts the chosen split (default: test) at each severity, runs Ultralytics
validation, and writes experiments/results/robustness_<tag>.json + .png. Nothing is invented: every number
comes from model.val().
  python -m evaluation.robustness --model models/best.pt --data datasets/prepared/gtsdb/data.yaml --split test
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import cv2
import yaml
from vision.corruptions import CORRUPTIONS, apply_corruption

from vision.paths import HOME

RESULTS = HOME / "experiments" / "results"
TMP = HOME / "experiments" / "tmp"


def build_corrupted(data_yaml: Path, split: str, corruption: str, severity: int, workdir: Path) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    img_dir, lbl_dir = root / cfg[split], root / cfg[split].replace("images", "labels")
    d = workdir / f"{corruption}_{severity}"
    (d / "images" / split).mkdir(parents=True, exist_ok=True); (d / "labels" / split).mkdir(parents=True, exist_ok=True)
    for p in img_dir.iterdir():
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(d / "images" / split / p.name), apply_corruption(img, corruption, severity))
        if (lbl_dir / f"{p.stem}.txt").exists():
            shutil.copy2(lbl_dir / f"{p.stem}.txt", d / "labels" / split / f"{p.stem}.txt")
    y = d / "data.yaml"
    y.write_text(yaml.safe_dump({"path": str(d.resolve()), split: f"images/{split}", "train": f"images/{split}",
                                 "val": f"images/{split}", "names": cfg["names"]}))
    return y


def run(model_path: str, data_yaml: Path, split: str, corruptions, severities, imgsz: int, device, tag: str, evaluate=None) -> dict:
    if evaluate is None:
        from ultralytics import YOLO
        model = YOLO(model_path)

        def evaluate(yaml_path: Path) -> dict:
            m = model.val(data=str(yaml_path), split=split, imgsz=imgsz, device=device, verbose=False, plots=False).box
            return {"precision": float(m.mp), "recall": float(m.mr), "map50": float(m.map50), "map50_95": float(m.map)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"model": model_path, "data": str(data_yaml), "split": split, "imgsz": imgsz, "results": {}, "simulated_corruptions": True}
    work = TMP / tag
    for c in corruptions:
        out["results"][c] = {}
        for s in severities:
            y = build_corrupted(data_yaml, split, c, s, work)
            out["results"][c][str(s)] = evaluate(y)
            print(c, s, out["results"][c][str(s)])
            shutil.rmtree(y.parent, ignore_errors=True)       # save disk
    (RESULTS / f"robustness_{tag}.json").write_text(json.dumps(out, indent=2))
    plot(out, RESULTS / f"robustness_{tag}.png")
    return out


def plot(out: dict, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, per in out["results"].items():
        xs = sorted(int(k) for k in per)
        ax.plot(xs, [per[str(x)]["map50"] for x in xs], marker="o", label=c)
    ax.set_xlabel("corruption severity (0 = clean)"); ax.set_ylabel("mAP@0.5"); ax.set_title("Robustness (measured)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="test"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None); ap.add_argument("--tag", default="baseline")
    ap.add_argument("--severities", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS))
    a = ap.parse_args()
    run(a.model, a.data, a.split, a.corruptions, a.severities, a.imgsz, a.device, a.tag)
