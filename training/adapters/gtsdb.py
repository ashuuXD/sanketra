"""GTSDB -> YOLO adapter.

Source: German Traffic Sign Detection Benchmark (FullIJCNN2013.zip: 900 PPM images 1360x800 + gt.txt).
Lines of gt.txt:  filename;leftCol;topRow;rightCol;bottomRow;classId   (class ids 0..42, same as GTSRB)
Check the dataset's own terms of use before using or sharing it.

Modes
  --mode category : 4 classes (prohibitory, danger, mandatory, other) using configs/gtsdb_categories.json (verify it!)
  --mode full     : all 43 sign classes (small data per class; expect weaker per-class results)
Images are renamed blkNNN_xxxxx so that training.prepare_dataset can split by block of neighbouring frames.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
FULL_NAMES = [
    "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60", "speed_limit_70", "speed_limit_80", "end_speed_limit_80",
    "speed_limit_100", "speed_limit_120", "no_overtaking", "no_overtaking_trucks", "right_of_way_next_junction", "priority_road",
    "give_way", "stop", "no_vehicles", "no_trucks", "no_entry", "general_danger", "curve_left", "curve_right", "double_curve",
    "bumpy_road", "slippery_road", "road_narrows_right", "road_works", "traffic_signals", "pedestrians", "children_crossing",
    "bicycles_crossing", "ice_snow", "wild_animals", "end_all_restrictions", "turn_right_ahead", "turn_left_ahead", "ahead_only",
    "ahead_or_right", "ahead_or_left", "keep_right", "keep_left", "roundabout", "end_no_overtaking", "end_no_overtaking_trucks"]


def load_categories(path: Path | None = None) -> tuple[list[str], dict[int, int]]:
    cfg = json.loads((path or ROOT / "configs" / "gtsdb_categories.json").read_text(encoding="utf-8"))
    names = cfg["names"]; lut = {}
    for i, n in enumerate(names[:-1]):
        for cid in cfg[n]:
            lut[cid] = i
    return names, lut


def convert(src: Path, out: Path, mode: str = "category", block: int = 10, include_empty: bool = True, cats: Path | None = None) -> dict:
    names, lut = (load_categories(cats) if mode == "category" else (FULL_NAMES, {}))
    other = len(names) - 1
    boxes = defaultdict(list)
    for line in (src / "gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        f, x1, y1, x2, y2, c = line.strip().split(";")
        boxes[f].append((int(x1), int(y1), int(x2), int(y2), int(c)))
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_empty = 0
    for p in sorted(src.glob("*.ppm")):
        bl = boxes.get(p.name, [])
        if not bl and not include_empty:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"WARNING: cannot read {p.name}"); continue
        h, w = img.shape[:2]
        stem = f"blk{int(p.stem) // block:03d}_{p.stem}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for x1, y1, x2, y2, c in bl:
            cls = lut.get(c, other) if mode == "category" else c
            lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        n_img += 1; n_box += len(lines); n_empty += (not lines)
    (out / "names.txt").write_text("\n".join(names))
    return {"images": n_img, "boxes": n_box, "empty_images": n_empty, "classes": len(names), "mode": mode}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder with *.ppm and gt.txt (the extracted FullIJCNN2013 folder)")
    ap.add_argument("--dst", type=Path, required=True); ap.add_argument("--mode", choices=["category", "full"], default="category")
    ap.add_argument("--block", type=int, default=10); ap.add_argument("--no-empty", action="store_true", help="drop images without sign boxes")
    a = ap.parse_args()
    print(convert(a.src, a.dst, a.mode, a.block, not a.no_empty))
    print("next: python -m training.prepare_dataset --src", a.dst, "--dst datasets/prepared/gtsdb --names", a.dst / "names.txt", r'--group-regex "^(blk\d+)_"')
