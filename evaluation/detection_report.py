"""Detection report on a held-out split: overall + per-class precision/recall/mAP and a confusion matrix.
  python -m evaluation.detection_report --model models/sanketra_gtsdb_n.pt --data datasets/prepared/gtsdb/data.yaml --tag gtsdb_n
Use the TEST split for numbers you report (the val split was used to pick the best epoch)."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

from vision.paths import HOME

RESULTS = HOME / "experiments" / "results"
TMP = HOME / "experiments" / "tmp"


def run(model_path: str, data_yaml: Path, split: str, imgsz: int, device, tag: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(model_path)
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, plots=True, verbose=False,
                  project=str(TMP), name=f"det_{tag}", exist_ok=True)
    box, names = m.box, model.names
    per_class = [{"class": names[int(c)], "precision": float(box.p[i]), "recall": float(box.r[i]),
                  "map50": float(box.ap50[i]), "map50_95": float(box.ap[i])} for i, c in enumerate(box.ap_class_index)]
    out = {"model": Path(model_path).name, "data": str(data_yaml), "split": split, "imgsz": imgsz,
           "overall": {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map)},
           "per_class": per_class, "confusion_png": None}
    RESULTS.mkdir(parents=True, exist_ok=True)
    cm = Path(m.save_dir) / "confusion_matrix_normalized.png"
    if cm.exists():
        shutil.copy2(cm, RESULTS / f"detection_{tag}_confusion.png"); out["confusion_png"] = f"detection_{tag}_confusion"
    (RESULTS / f"detection_{tag}.json").write_text(json.dumps(out, indent=2))
    shutil.rmtree(Path(m.save_dir), ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="test"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None); ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    o = run(a.model, a.data, a.split, a.imgsz, a.device, a.tag)
    print(json.dumps(o["overall"], indent=2))
    for r in o["per_class"]:
        print(f"{r['class']:14} P {r['precision']:.3f}  R {r['recall']:.3f}  mAP50 {r['map50']:.3f}  mAP50-95 {r['map50_95']:.3f}")
