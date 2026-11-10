"""Validate + split a YOLO-format dataset (images/ + labels/ flat folders) with group-level splitting.

Why group-level: frames from one sequence/location are near-duplicates; splitting them across
train/val/test leaks information and inflates metrics.

Usage:
  python -m training.prepare_dataset --src datasets/raw/mydata --dst datasets/prepared/mydata \
      --names names.txt --group-regex "^(.*?)_\\d+$" --seed 0
"""
from __future__ import annotations
import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def ahash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)
    bits = (g > g.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def read_labels(path: Path, n_classes: int) -> tuple[list[tuple], list[str]]:
    rows, errs = [], []
    if not path.exists():
        return rows, ["missing label file"]
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        p = line.split()
        try:
            c, x, y, w, h = int(p[0]), *(float(v) for v in p[1:5])
            assert len(p) == 5 and 0 <= c < n_classes and all(0 <= v <= 1 for v in (x, y, w, h)) and w > 0 and h > 0
            rows.append((c, x, y, w, h))
        except Exception:
            errs.append(f"line {i} invalid: '{line}'")
    return rows, errs


def prepare(src: Path, dst: Path, names: list[str], group_regex: str | None, seed: int = 0,
            ratios=(0.7, 0.15, 0.15), dup_hamming: int = 2) -> dict:
    img_dir, lbl_dir = src / "images", src / "labels"
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    rx = re.compile(group_regex) if group_regex else None
    good, report = [], {"corrupt_images": [], "bad_labels": {}, "empty_labels": 0, "duplicates": []}
    hashes: list[tuple[int, str]] = []
    group_of: dict[str, str] = {}
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            report["corrupt_images"].append(p.name); continue
        rows, errs = read_labels(lbl_dir / f"{p.stem}.txt", len(names))
        if errs and not rows:
            report["bad_labels"][p.name] = errs; continue
        if not rows:
            report["empty_labels"] += 1
        h = ahash(img)
        dup = next((n for hh, n in hashes if bin(hh ^ h).count("1") <= dup_hamming), None)
        if dup:
            report["duplicates"].append([p.name, dup])   # near-duplicate: kept but forced into the same group
        hashes.append((h, p.name))
        group = rx.match(p.stem).group(1) if rx and rx.match(p.stem) else p.stem
        if dup:                       # near-duplicates must share a group so they cannot straddle splits
            group = group_of[dup]
        group_of[p.name] = group
        good.append((group, p.name, rows))
    groups: dict[str, list] = defaultdict(list)
    for g, n, r in good:
        groups[g].append((n, r))
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    total, split_of, acc = len(good), {}, 0
    cut = (ratios[0] * total, (ratios[0] + ratios[1]) * total)
    for k in keys:
        s = "train" if acc < cut[0] else "val" if acc < cut[1] else "test"
        split_of[k] = s; acc += len(groups[k])
    if dst.exists():
        shutil.rmtree(dst)
    stats = {s: {"images": 0, "instances": Counter()} for s in ("train", "val", "test")}
    for k, items in groups.items():
        s = split_of[k]
        for n, rows in items:
            (dst / "images" / s).mkdir(parents=True, exist_ok=True); (dst / "labels" / s).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_dir / n, dst / "images" / s / n)
            (dst / "labels" / s / f"{Path(n).stem}.txt").write_text("\n".join(" ".join(map(str, r)) for r in rows))
            stats[s]["images"] += 1
            stats[s]["instances"].update(names[r[0]] for r in rows)
    out = {s: {"images": v["images"], "instances": dict(v["instances"])} for s, v in stats.items()}
    summary = {"source": str(src), "seed": seed, "ratios": ratios, "group_regex": group_regex, "num_groups": len(groups),
               "splits": out, "quality": {**report, "n_corrupt": len(report["corrupt_images"]),
               "n_bad_labels": len(report["bad_labels"]), "n_duplicates": len(report["duplicates"])}}
    (dst / "data.yaml").write_text(yaml.safe_dump({"path": str(dst.resolve()), "train": "images/train", "val": "images/val",
                                                   "test": "images/test", "names": dict(enumerate(names))}))
    (dst / "dataset_stats.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path); ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--names", required=True, type=Path, help="text file, one class name per line")
    ap.add_argument("--group-regex", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = [n.strip() for n in a.names.read_text().splitlines() if n.strip()]
    s = prepare(a.src, a.dst, names, a.group_regex, a.seed)
    print(json.dumps({k: s[k] for k in ("num_groups", "splits")}, indent=2)); print("quality:", s["quality"]["n_corrupt"], "corrupt,",
          s["quality"]["n_bad_labels"], "bad labels,", s["quality"]["n_duplicates"], "near-duplicates")
