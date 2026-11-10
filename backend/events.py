"""Bounded, thread-safe detection event log with per-event frame + thumbnail."""
from __future__ import annotations
import base64
import csv
import io
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
import cv2
import numpy as np
from vision.calibration import iou

EXPORT_FIELDS = ["id", "session", "track_id", "class_name", "label_en", "category", "first_seen",
                 "last_seen", "hits", "best_conf", "last_conf", "uncertain", "source"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_thumb(frame: np.ndarray, box, size: int = 72) -> str | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.resize(frame[y1:y2, x1:x2], (size, size)), [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode() if ok else None


class EventLog:
    def __init__(self, max_events: int = 200):
        self.max_events, self.lock = max_events, threading.Lock()
        self.events: "OrderedDict[int, dict]" = OrderedDict()
        self.frames: dict[int, bytes] = {}
        self.thumbs: dict[int, str | None] = {}
        self.by_track: dict[tuple, int] = {}
        self._state: dict[int, dict] = {}   # last box/time per event (for untracked association)
        self._next_id, self.session = 1, 0

    def new_session(self) -> int:
        with self.lock:
            self.session += 1
            return self.session

    def _evict(self):
        while len(self.events) > self.max_events:
            eid, ev = self.events.popitem(last=False)
            self.frames.pop(eid, None)
            self.thumbs.pop(eid, None)
            self.by_track.pop((ev["session"], ev["track_id"]), None)
            self._state.pop(eid, None)

    def update(self, dets: list[dict], raw_frame: np.ndarray, annotated_jpeg: bytes, source: str) -> None:
        with self.lock:
            now = time.monotonic()
            for d in dets:
                key = (self.session, d["track_id"]) if d["track_id"] is not None else None
                eid = self.by_track.get(key) if key else self._match_untracked(d, source, now)
                if eid is not None and eid in self.events:
                    ev = self.events[eid]
                    ev["last_seen"], ev["hits"], ev["last_conf"] = _now(), ev["hits"] + 1, d["conf"]
                    ev["uncertain"] = d["uncertain"]
                    self._state[eid] = {"t": now, "box": d["box"]}
                    if d["conf"] > ev["best_conf"] + 0.05:
                        ev["best_conf"] = d["conf"]
                        self.frames[eid], self.thumbs[eid] = annotated_jpeg, make_thumb(raw_frame, d["box"])
                    continue
                eid, self._next_id = self._next_id, self._next_id + 1
                self.events[eid] = {"id": eid, "session": self.session, "track_id": d["track_id"],
                                    "class_name": d["class_name"], "label_en": d["label_en"],
                                    "category": d["category"], "first_seen": _now(), "last_seen": _now(),
                                    "hits": 1, "best_conf": d["conf"], "last_conf": d["conf"],
                                    "uncertain": d["uncertain"], "source": source}
                self.frames[eid], self.thumbs[eid] = annotated_jpeg, make_thumb(raw_frame, d["box"])
                self._state[eid] = {"t": now, "box": d["box"]}
                if key:
                    self.by_track[key] = eid
                self._evict()

    def _match_untracked(self, d: dict, source: str, now: float):
        """Tracking off: merge a detection into a recent event of the same class at a similar place,
        otherwise every video frame would create a new event. Still images are always new events."""
        if source == "image" or source.startswith("lab"):
            return None
        for eid in list(reversed(self.events))[:50]:
            ev = self.events[eid]
            st = self._state.get(eid)
            if (ev["session"] == self.session and ev["track_id"] is None and ev["class_name"] == d["class_name"]
                    and st and now - st["t"] <= 1.5 and iou(st["box"], d["box"]) >= 0.3):
                return eid
        return None

    def list(self, class_name: str | None = None, limit: int = 100) -> list[dict]:
        with self.lock:
            evs = [dict(e, thumb=self.thumbs.get(e["id"])) for e in reversed(self.events.values())
                   if class_name in (None, "", e["class_name"])]
        return evs[:limit]

    def frame(self, eid: int) -> bytes | None:
        with self.lock:
            return self.frames.get(eid)

    def clear(self):
        with self.lock:
            self.events.clear(); self.frames.clear(); self.thumbs.clear(); self.by_track.clear(); self._state.clear()

    def session_unique(self) -> int:
        with self.lock:
            return sum(1 for e in self.events.values() if e["session"] == self.session)

    def export(self, fmt: str) -> tuple[str, str]:
        with self.lock:
            rows = [{k: e[k] for k in EXPORT_FIELDS} for e in self.events.values()]
        if fmt == "json":
            import json
            return json.dumps(rows, indent=2), "application/json"
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS)
        w.writeheader(); w.writerows(rows)
        return buf.getvalue(), "text/csv"
