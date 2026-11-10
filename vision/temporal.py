"""Temporal consistency helpers: per-track EMA of confidence and a flicker metric."""
from __future__ import annotations


class ConfidenceSmoother:
    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha, self.state = alpha, {}

    def update(self, track_id, conf: float) -> float:
        prev = self.state.get(track_id)
        val = conf if prev is None else self.alpha * conf + (1 - self.alpha) * prev
        self.state[track_id] = val
        return val

    def reset(self):
        self.state.clear()


def flicker_rate(presence: list[bool]) -> float:
    """Fraction of consecutive frame pairs where presence flips (lower = more stable).
    presence: per-frame booleans for one object/track between its first and last sighting."""
    if len(presence) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(presence, presence[1:]) if a != b)
    return flips / (len(presence) - 1)
