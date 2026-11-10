"""Temporal consistency (RoadSight RQ3): does smoothing over time reduce flickering detections?

Runs the detector + tracker over a video and, for every track, compares three display rules:
  raw        : show a box when this frame's confidence >= threshold
  ema        : show a box when the exponential moving average of the track's confidence >= threshold
  ema_hold3  : like ema, and keep showing the box through up to 3 missed frames
Flicker rate = fraction of consecutive frames where a track's displayed/not-displayed state flips (lower = more stable).
No ground truth is needed, so this measures STABILITY, not accuracy: a stable wrong detection also scores well.
  python -m evaluation.temporal_eval --model models/sanketra_gtsdb_n.pt --video my_road_video.mp4 --tag gtsdb_n
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from vision.paths import HOME
from vision.temporal import flicker_rate

RESULTS = HOME / "experiments" / "results"


def track_states(obs: list[tuple[int, float]], thr: float, alpha: float, hold: int):
    """obs: [(frame_idx, conf)] for one track. Returns per-frame booleans (raw, ema, ema_hold) across the track's span."""
    f0, f1 = obs[0][0], obs[-1][0]
    by = dict(obs)
    raw, ema_s, hold_s = [], [], []
    ema, held, last = None, 0, False
    for f in range(f0, f1 + 1):
        if f in by:
            c = by[f]
            ema = c if ema is None else alpha * c + (1 - alpha) * ema
            raw.append(c >= thr); show = ema >= thr; ema_s.append(show); hold_s.append(show); last, held = show, 0
        else:
            raw.append(False); ema_s.append(False); held += 1
            hold_s.append(last and held <= hold)
    return raw, ema_s, hold_s


def run(model_path: str, video: str, imgsz: int, conf_low: float, thr: float, alpha: float, max_frames: int, tag: str, device=None) -> dict:
    import cv2
    from ultralytics import YOLO
    model = YOLO(model_path); cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video}")
    tracks: dict[int, list[tuple[int, float]]] = defaultdict(list); n = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        r = model.track(fr, persist=True, tracker="bytetrack.yaml", conf=conf_low, imgsz=imgsz, device=device, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            for tid, c in zip(r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
                tracks[tid].append((n, float(c)))
        n += 1
    cap.release()
    rows = {"raw": [], "ema": [], "ema_hold3": []}; lens = []
    for obs in tracks.values():
        if len(obs) < 10:
            continue
        lens.append(obs[-1][0] - obs[0][0] + 1)
        for k, states in zip(rows, track_states(obs, thr, alpha, 3)):
            rows[k].append(flicker_rate(states))
    out = {"model": Path(model_path).name, "video": Path(video).name, "frames": n, "imgsz": imgsz, "detector_conf_floor": conf_low,
           "display_threshold": thr, "ema_alpha": alpha, "tracks_analysed": len(lens),
           "mean_track_span_frames": round(float(np.mean(lens)), 1) if lens else None,
           "flicker_rate": {k: (round(float(np.mean(v)), 4) if v else None) for k, v in rows.items()}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"temporal_{tag}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--conf-floor", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=600); ap.add_argument("--tag", default="baseline"); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.model, a.video, a.imgsz, a.conf_floor, a.threshold, a.alpha, a.max_frames, a.tag, a.device), indent=2))
