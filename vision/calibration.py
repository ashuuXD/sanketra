"""Confidence calibration utilities for detector scores (sigmoid class scores).

Temperature scaling on the logit of the confidence: p' = sigmoid(logit(p) / T).
T>1 softens over-confident scores, T<1 sharpens. Fitted on a held-out split only.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def apply_temperature(conf, T: float):
    if T <= 0:
        raise ValueError("temperature must be > 0")
    return 1.0 / (1.0 + np.exp(-_logit(conf) / T))


def nll(conf, correct) -> float:
    p = np.clip(np.asarray(conf, dtype=np.float64), _EPS, 1 - _EPS)
    y = np.asarray(correct, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(conf, correct, grid=None) -> float:
    """Grid search T minimising NLL. conf: scores, correct: 1 if true positive else 0."""
    conf, correct = np.asarray(conf), np.asarray(correct)
    if len(conf) < 10 or correct.min() == correct.max():
        raise ValueError("need >=10 samples containing both true and false positives to fit T")
    grid = np.linspace(0.3, 5.0, 236) if grid is None else grid
    losses = [nll(apply_temperature(conf, t), correct) for t in grid]
    return float(grid[int(np.argmin(losses))])


def ece(conf, correct, n_bins: int = 10):
    """Expected Calibration Error + per-bin stats for a reliability diagram."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    total, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        total += m.sum() / len(conf) * abs(c - a)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": c, "acc": a})
    return float(total), bins


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_predictions(preds, gts, iou_thr: float = 0.5):
    """Greedy matching. preds: list of (cls, conf, box); gts: list of (cls, box).
    Returns (conf array, correct array, n_missed_gt)."""
    used, confs, correct = set(), [], []
    for cls, conf, box in sorted(preds, key=lambda p: -p[1]):
        best, best_i = 0.0, -1
        for i, (gc, gb) in enumerate(gts):
            if i in used or gc != cls:
                continue
            v = iou(box, gb)
            if v > best:
                best, best_i = v, i
        ok = best >= iou_thr
        if ok:
            used.add(best_i)
        confs.append(conf)
        correct.append(1 if ok else 0)
    return np.array(confs), np.array(correct), len(gts) - len(used)
