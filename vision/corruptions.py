"""Controlled, seeded image corruptions for robustness experiments (severity 0..5)."""
from __future__ import annotations
import cv2
import numpy as np

CORRUPTIONS = ("brightness", "blur", "noise", "lowres", "lowlight")
MAX_SEVERITY = 5


def _check(img: np.ndarray, severity: int) -> None:
    if img is None or img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("expected a HxWx3 uint8 BGR image")
    if not 0 <= severity <= MAX_SEVERITY:
        raise ValueError(f"severity must be in [0, {MAX_SEVERITY}]")


def apply_corruption(img: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    """Return a corrupted copy. severity 0 returns an unchanged copy."""
    _check(img, severity)
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption '{name}', choose from {CORRUPTIONS}")
    if severity == 0:
        return img.copy()
    s = severity
    if name == "brightness":      # overexposure
        out = img.astype(np.float32) * (1 + 0.25 * s)
    elif name == "lowlight":      # darkening + gamma, closer to dusk/night
        out = 255.0 * (img.astype(np.float32) / 255.0) ** (1 + 0.5 * s) * (1 - 0.1 * s)
    elif name == "blur":          # motion-like horizontal blur
        k = 2 * s + 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(img, -1, kernel).astype(np.float32)
    elif name == "noise":
        rng = np.random.default_rng(seed)
        out = img.astype(np.float32) + rng.normal(0, 8 * s, img.shape)
    else:                         # lowres: downscale then upscale back
        h, w = img.shape[:2]
        f = 1.0 / (1 + s)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
