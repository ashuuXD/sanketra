"""Create a short 16:9 demo clip by panning a window over a still image (no webcam or dataset needed).
This is a SIMULATED clip for UI testing, not real driving footage.
  python scripts/make_demo_video.py --image tests/assets/street.jpg --out data/uploads/demo_pan.mp4
"""
import argparse
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.paths import HOME

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="tests/assets/street.jpg"); ap.add_argument("--out", default=str(HOME / "data" / "uploads" / "demo_pan.mp4"))
ap.add_argument("--seconds", type=int, default=8); ap.add_argument("--fps", type=int, default=15); ap.add_argument("--width", type=int, default=960)
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
img = cv2.imread(a.image)
if img is None:
    raise SystemExit(f"cannot read {a.image}")
H, W = img.shape[:2]
OW, OH = a.width, a.width * 9 // 16
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OW, OH))
n = a.seconds * a.fps
for i in range(n):
    t = i / (n - 1)
    zoom = 1.0 + 0.2 * (1 - abs(2 * t - 1))          # gentle zoom in then out
    cw = int(W / zoom); ch = int(cw * 9 / 16)
    x = (W - cw) // 2
    y = int((H - ch) * t)                            # pan from top to bottom of the image
    vw.write(cv2.resize(img[y:y + ch, x:x + cw], (OW, OH)))
vw.release(); print("wrote", a.out, f"({OW}x{OH}, {a.seconds}s)")
