"""Overlay rendering (OpenCV, BGR colours)."""
from __future__ import annotations
import cv2
import numpy as np

CATEGORY_COLOURS = {  # BGR; red is deliberately not used for normal detections
    "regulatory": (255, 214, 0), "warning": (0, 191, 255), "mandatory": (166, 200, 0),
    "information": (200, 160, 120), "object": (200, 170, 140), "other": (190, 175, 165), "signal": (200, 200, 200), "unknown": (170, 170, 170),
}
UNCERTAIN = (0, 165, 255)


def _dashed_rect(img, p1, p2, colour, t=2, dash=8):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, t)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, t)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, t)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, t)


def draw_detections(frame: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        colour = UNCERTAIN if d["uncertain"] else CATEGORY_COLOURS.get(d["category"], CATEGORY_COLOURS["unknown"])
        if d["uncertain"]:
            _dashed_rect(out, (x1, y1), (x2, y2), colour)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        tid = f"#{d['track_id'] % 100000} " if d.get("track_id") is not None else ""
        text = f"{tid}{d['label_en']} {d['conf']:.2f}" + (" ?" if d["uncertain"] else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 8)
        cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty), colour, -1)
        cv2.putText(out, text, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out
