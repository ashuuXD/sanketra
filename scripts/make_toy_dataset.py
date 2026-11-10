"""Generate a small SYNTHETIC sign-like dataset (circles, triangles, discs on random backgrounds) in YOLO format.
Purpose: prove the whole prepare -> train -> evaluate pipeline runs on your machine in minutes. These are NOT real signs
and any metric computed from it says nothing about real-world performance.
  python scripts/make_toy_dataset.py --dst datasets/raw/toy --n 90
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--dst", type=Path, default=Path("datasets/raw/toy")); ap.add_argument("--n", type=int, default=90)
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed); NAMES = ["red_ring", "red_triangle", "blue_disc"]
(a.dst / "images").mkdir(parents=True, exist_ok=True); (a.dst / "labels").mkdir(parents=True, exist_ok=True)
for i in range(a.n):
    W, H = 640, 384
    bg = cv2.GaussianBlur((rng.random((H // 8, W // 8, 3)) * 160 + 40).astype(np.uint8), (3, 3), 0)
    img = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC); labels = []
    for _ in range(rng.integers(1, 4)):
        c = int(rng.integers(0, 3)); r = int(rng.integers(18, 48)); x = int(rng.integers(r + 4, W - r - 4)); y = int(rng.integers(r + 4, H - r - 4))
        if c == 0:
            cv2.circle(img, (x, y), r, (30, 30, 220), -1); cv2.circle(img, (x, y), int(r * 0.68), (245, 245, 245), -1)
        elif c == 1:
            pts = np.array([[x, y - r], [x - r, y + r], [x + r, y + r]]); cv2.fillPoly(img, [pts], (30, 30, 220))
            q = np.array([[x, y - r * 0.45], [x - r * 0.5, y + r * 0.72], [x + r * 0.5, y + r * 0.72]], np.int32); cv2.fillPoly(img, [q], (245, 245, 245))
        else:
            cv2.circle(img, (x, y), r, (200, 90, 20), -1); cv2.circle(img, (x, y), int(r * 0.45), (245, 245, 245), -1)
        labels.append(f"{c} {x / W:.6f} {y / H:.6f} {2 * r / W:.6f} {2 * r / H:.6f}")
    stem = f"seq{i // 6:03d}_{i:04d}"
    cv2.imwrite(str(a.dst / "images" / f"{stem}.jpg"), img); (a.dst / "labels" / f"{stem}.txt").write_text("\n".join(labels))
(a.dst / "names.txt").write_text("\n".join(NAMES)); print("wrote", a.n, "synthetic images to", a.dst)
