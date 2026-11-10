"""Inference engine: model loading, detection, tracking, calibration, smoothing, latency stats."""
from __future__ import annotations
import logging
import threading
import time
from collections import deque
from pathlib import Path
import numpy as np
from vision.calibration import apply_temperature
from vision.temporal import ConfidenceSmoother
from .registry import SignRegistry
from .settings import HOME, ROOT, RuntimeSettings

log = logging.getLogger("sanketra.engine")


def resolve_model_path(cfg_path: str, models_dir: Path | None = None) -> Path:
    """'auto' = newest sanketra_*.pt in your data folder (SanketraData\\models) or the project's models folder,
    else the generic yolo11n.pt. A plain file name is looked up in the same two folders."""
    dirs = [models_dir] if models_dir else [HOME / "models", ROOT / "models"]
    if cfg_path == "auto":
        found = sorted((p for d in dirs for p in d.glob("sanketra_*.pt")), key=lambda p: p.stat().st_mtime)
        return found[-1] if found else Path("yolo11n.pt")
    path = Path(cfg_path)
    if path.is_absolute():
        return path
    for d in dirs:
        if (d / path).exists():
            return d / path
    return path


class Engine:
    def __init__(self, cfg: dict, registry: SignRegistry):
        self.cfg, self.registry = cfg, registry
        self.model = None
        self.status, self.error = "uninitialized", None
        self.device = "cpu"
        self.lock = threading.Lock()
        self.latencies: deque[float] = deque(maxlen=300)
        self.smoother = ConfidenceSmoother(cfg["inference"]["smoothing_alpha"])
        self.class_ids: list[int] | None = None
        self.prec: dict = {}
        self.aux = None                # optional second, general-purpose detector (people and vehicles)
        self.aux_ids: list[int] | None = None
        self._aux_n = 0
        self._aux_last: list[dict] = []
        self._skip = 0                 # first frames after a stream start / setting change are warm-up, not statistics

    # ---- lifecycle -------------------------------------------------
    def load(self) -> None:
        self.status, self.error = "loading", None
        try:
            import torch
            from ultralytics import YOLO
            want = self.cfg["model"]["device"]
            self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if want == "auto" else want
            path = resolve_model_path(self.cfg["model"]["path"])
            self.model_path = str(path)
            self.model = YOLO(self.model_path)
            names = self.model.names
            flt = self.cfg["model"].get("class_filter")
            self.class_ids = [i for i, n in names.items() if n in flt] if flt else None
            t0 = time.perf_counter()
            self.prec = self._precision_kw()
            # One cheap warm-up call only. (cudnn.benchmark / warming every size made startup minutes long on a 4 GB GPU.)
            self.model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, device=self.device, verbose=False, **self.prec, **self.shape_kw())
            self.model.predictor = None
            self._warm_tracker()
            self._load_aux(YOLO)
            log.info("warm-up took %.1fs (precision args: %s)", time.perf_counter() - t0, self.prec or "fp32")
            self.status = "ready"
            log.info("model loaded: %s on %s", self.model_path, self.device)
        except Exception as e:  # never silent: surface in /api/health
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            log.exception("model load failed")

    AUX_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

    def _load_aux(self, YOLO) -> None:
        """Second detector so the console also shows people and vehicles next to signs (only when the main model is sign-specific)."""
        self.aux, self.aux_ids = None, None
        if len(self.model.names) == 80:          # main model is already the general COCO model
            return
        try:
            cands = [HOME / "models" / "yolo11n.pt", ROOT / "models" / "yolo11n.pt", Path("yolo11n.pt")]
            found = next((c for c in cands if c.exists()), None)
            aux = YOLO(str(found) if found else "yolo11n.pt")      # downloads it once if it is not on disk
            self.aux_ids = [i for i, n in aux.names.items() if n in self.AUX_CLASSES]
            aux.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, device=self.device, verbose=False, classes=self.aux_ids)
            aux.predictor = None
            self.aux = aux
            log.info("objects model ready (people and vehicles)")
        except Exception as e:                      # optional feature: never fatal
            self.aux = None
            log.warning("objects model unavailable: %s", e)

    def _precision_kw(self) -> dict:
        """FP16 on CUDA, using whichever argument name this Ultralytics version expects ('half' is deprecated)."""
        if not self.device.startswith("cuda") or not self.cfg["model"].get("fp16", False):
            return {}
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
        except Exception:
            return {}

    def _warm_tracker(self) -> None:
        """The first tracker update that has real detections lazily imports matching code (about 5 s on Windows).
        Do it now, on a sample image, so pressing Play is instant."""
        try:
            import cv2
            img = cv2.imread(str(ROOT / "data" / "samples" / "street_bus.jpg"))
            if img is None:
                return
            for _ in range(2):
                self.model.track(img, persist=True, tracker="bytetrack.yaml", imgsz=640, device=self.device,
                                 verbose=False, **self.prec, **self.shape_kw())
        except Exception as e:                      # warm-up is best-effort, never fatal
            log.warning("tracker warm-up skipped: %s", e)
        finally:
            self.model.predictor = None

    def _sidecar(self) -> dict:
        """Optional <weights>.json written by the training script: what the model was trained on."""
        try:
            import json
            return json.loads(Path(self.model_path).with_suffix(".json").read_text(encoding="utf-8-sig"))
        except Exception:
            pass
        if self.model is not None and set(self.model.names.values()) == {"prohibitory", "danger", "mandatory", "other"}:
            return {"note": "Trained on GTSDB (German signs). Detects sign GROUPS (prohibitory, danger, mandatory, other). It does not recognise Indian signs."}
        return {}

    def shape_kw(self) -> dict:
        """fixed_input_shape: letterbox every source to one square size, so the GPU sees ONE input shape
        (a new shape can cost seconds on first use). Turn off in configs/default.yaml for rectangular inference."""
        return {"rect": False} if self.cfg["inference"].get("fixed_input_shape", True) else {}

    def info(self) -> dict:
        if self.model is None:
            return {"status": self.status, "error": self.error}
        p = Path(self.model_path)
        params = sum(x.numel() for x in self.model.model.parameters())
        names = self.model.names
        return {"status": self.status, "error": self.error, "name": p.name, "device": self.device,
                "params_million": round(params / 1e6, 2),
                "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None,
                "num_classes": len(names), "classes": list(names.values()),
                "is_coco_demo_model": len(names) == 80,
                "training_note": self._sidecar().get("note"),
                "objects_available": self.aux is not None,
                "registry_version": self.registry.version}

    def note_change(self, frames: int = 3) -> None:
        """A setting that changes the GPU input shape was changed: the next frames may be slow once."""
        self._skip = frames

    def reset_tracking(self) -> None:
        with self.lock:
            if self.model is not None:
                self.model.predictor = None  # forces a fresh tracker
            if self.aux is not None:
                self.aux.predictor = None
            self._aux_n, self._aux_last = 0, []
            self.smoother.reset()
            self.latencies.clear()      # stats should describe the current stream, not earlier calls
            self._skip = 2

    # ---- inference -------------------------------------------------
    def infer(self, frame: np.ndarray, rs: RuntimeSettings, track: bool) -> tuple[list[dict], float]:
        if self.status != "ready":
            raise RuntimeError(f"model not ready (status={self.status}, error={self.error})")
        t0 = time.perf_counter()
        kw = dict(conf=rs.conf, imgsz=rs.imgsz, device=self.device, verbose=False, classes=self.class_ids, **self.prec, **self.shape_kw())
        with self.lock:
            if track and rs.tracking:
                res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)[0]
            else:
                res = self.model.predict(frame, **kw)[0]
            res_aux = None
            run_aux = False
            if rs.objects and self.aux is not None:
                every = max(1, int(self.cfg["inference"].get("objects_every", 2)))
                run_aux = (self._aux_n % every == 0)
                self._aux_n += 1
                if run_aux:
                    akw = dict(conf=max(rs.conf, 0.3), imgsz=min(rs.imgsz, 416), device=self.device, verbose=False, classes=self.aux_ids)
                    if track and rs.tracking:
                        res_aux = self.aux.track(frame, persist=True, tracker="bytetrack.yaml", **akw)[0]
                    else:
                        res_aux = self.aux.predict(frame, **akw)[0]
            else:
                self._aux_last = []
        latency = (time.perf_counter() - t0) * 1000
        if self._skip > 0:
            self._skip -= 1
            log.info("warm-up inference: frame %dx%d, imgsz %d, took %.0f ms (not counted in statistics)", frame.shape[1], frame.shape[0], rs.imgsz, latency)
        else:
            self.latencies.append(latency)
        if latency > 1500:
            sp = getattr(res, "speed", None) or {}
            model_ms = sum(sp.get(k, 0) for k in ("preprocess", "inference", "postprocess"))
            log.warning("slow inference %.0f ms (frame %dx%d, imgsz %d): ultralytics preprocess %.0f / model %.0f / postprocess %.0f ms; "
                        "the other %.0f ms was tracker, transfers or waiting", latency, frame.shape[1], frame.shape[0], rs.imgsz,
                        sp.get("preprocess", 0), sp.get("inference", 0), sp.get("postprocess", 0), latency - model_ms)
        dets = self._to_dets(res, rs)
        if res_aux is not None:
            self._aux_last = self._to_dets(res_aux, rs, aux=True)
        dets += self._aux_last          # on skipped frames the previous people/vehicle boxes are reused
        return dets, latency

    OBJ_ID_OFFSET = 100000       # keeps track ids of the two detectors apart

    def _to_dets(self, res, rs: RuntimeSettings, aux: bool = False) -> list[dict]:
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
        for box, c, k, tid in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist(),
                                  res.boxes.cls.int().cpu().tolist(), ids):
            name = res.names[k]
            if aux:
                reg = {"label_en": name.title(), "category": "object"}
                tid = tid + self.OBJ_ID_OFFSET if tid is not None else None
            else:
                reg = self.registry.lookup(name)
            conf = float(apply_temperature(c, rs.temperature))
            if rs.smoothing and tid is not None:
                conf = self.smoother.update(tid, conf)
            out.append({"track_id": tid, "class_id": k, "class_name": name, "label_en": reg["label_en"],
                        "category": reg["category"], "raw_conf": round(c, 4), "conf": round(conf, 4),
                        "uncertain": conf < rs.abstain_below, "box": [round(v, 1) for v in box]})
        return out

    def latency_stats(self) -> dict:
        if not self.latencies:
            return {"n": 0}
        a = np.array(list(self.latencies)[-150:])
        return {"n": int(a.size), "median_ms": round(float(np.median(a)), 1),
                "p95_ms": round(float(np.percentile(a, 95)), 1) if a.size >= 20 else None,   # a p95 of a few samples is meaningless
                "last_ms": round(float(a[-1]), 1)}
