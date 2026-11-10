"""Fine-tune a YOLO model. Config is saved next to the run for reproducibility.
Local 4 GB GPU: batch 8, imgsz 640 usually fits for yolo11n. Use Colab for yolo11s or larger batches.
  python -m training.train --data datasets/prepared/gtsdb/data.yaml --model yolo11n.pt --epochs 100 --name gtsdb_n
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--name", default="exp")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    # fliplr=0: left/right-handed signs (keep-left, turn-left) change meaning when mirrored.
    from vision.paths import HOME
    project = HOME / "experiments" / "runs"   # absolute: Ultralytics nests relative paths
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, device=a.device,
                project=str(project), name=a.name, fliplr=0.0, patience=30, deterministic=True, workers=2, exist_ok=True)
    out = Path(model.trainer.save_dir)                 # the real output folder, whatever the Ultralytics version does
    (out / "sanketra_train_config.json").write_text(json.dumps(vars(a), indent=2))
    best = out / "weights" / "best.pt"
    print("best weights:", best, "(exists)" if best.exists() else "(MISSING)")


if __name__ == "__main__":
    main()
