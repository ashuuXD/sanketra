"""Sanketra API. Run: uvicorn backend.main:app --port 8000"""
from __future__ import annotations
import asyncio
import base64
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from vision.corruptions import CORRUPTIONS, apply_corruption
from vision.draw import draw_detections
from .engine import Engine
from .events import EventLog
from .registry import SignRegistry
from .settings import DISCLAIMER, HOME, ROOT, RuntimeSettings, load_config

logging.basicConfig(level=logging.INFO, format='{"t":"%(asctime)s","lvl":"%(levelname)s","src":"%(name)s","msg":"%(message)s"}')
log = logging.getLogger("sanketra.api")

UPLOADS = HOME / "data" / "uploads"
VIDEO_EXT, IMAGE_EXT = {".mp4", ".avi", ".mov", ".mkv"}, {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class State:
    def __init__(self):
        self.cfg = load_config()
        self.registry = SignRegistry()
        self.engine = Engine(self.cfg, self.registry)
        self.events = EventLog(self.cfg["events"]["max_events"])
        self.rs = RuntimeSettings.from_config(self.cfg)
        self.stream_active, self.stream_stop = False, False
        self.stream_stats: dict = {}


S = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    asyncio.get_running_loop().run_in_executor(None, S.engine.load)  # load without blocking startup
    yield
    S.stream_stop = True


app = FastAPI(title="Sanketra", version="0.1.0", lifespan=lifespan)


def encode_jpeg(img: np.ndarray, max_w: int | None = None, quality: int | None = None) -> bytes:
    max_w = max_w or S.cfg["stream"]["max_width"]
    quality = quality or S.cfg["stream"]["jpeg_quality"]
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "File is not a decodable image (use jpg/png/bmp/webp).")
    return img


async def read_upload(file: UploadFile, allowed: set[str]) -> bytes:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}")
    data = await file.read()
    if len(data) > S.cfg["limits"]["max_upload_mb"] * 1e6:
        raise HTTPException(413, "File too large.")
    return data


def require_ready():
    if S.engine.status != "ready":
        raise HTTPException(503, f"Model not ready: status={S.engine.status} error={S.engine.error}")


# ------------------------------ REST ------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "model_status": S.engine.status, "model_error": S.engine.error,
            "version": app.version, "ui_build": ui_build_id(), "disclaimer": DISCLAIMER}


@app.get("/api/model")
def model_info():
    return {**S.engine.info(), "settings": S.rs.model_dump(), "latency": S.engine.latency_stats()}


@app.get("/api/settings")
def get_settings():
    return S.rs.model_dump()


@app.post("/api/settings")
def set_settings(body: dict):
    try:
        before = S.rs.imgsz
        S.rs = RuntimeSettings(**{**S.rs.model_dump(), **body})
    except (ValidationError, ValueError) as e:
        raise HTTPException(422, f"Invalid settings: {e}")
    if S.rs.imgsz != before:
        S.engine.note_change()
    return S.rs.model_dump()


@app.get("/api/registry")
def registry():
    """Registry entries, each flagged with whether the ACTIVE model can detect it (so the Signs page shows what is done per sign)."""
    names = set(S.engine.model.names.values()) if S.engine.model is not None else set()
    signs = [{**x, "in_model": x["class_name"] in names, "needs_review": bool(x.get("needs_review", False))} for x in S.registry.all()]
    return {"version": S.registry.version, "note": S.registry.note, "signs": signs, "model_classes": sorted(names)}


@app.get("/api/status")
def project_status():
    """What is done and what is not, computed from the real state of this installation (model, results files, registry)."""
    res = {p.stem for p in (HOME / "experiments" / "results").glob("*.json")}
    has = lambda prefix: any(r.startswith(prefix) for r in res)
    names = set(S.engine.model.names.values()) if S.engine.model is not None else set()
    reg = {x["class_name"] for x in S.registry.all()}
    gtsdb = {"prohibitory", "danger", "mandatory", "other"}
    indian = (names - gtsdb) & reg if len(names) != 80 else set()
    measured = [has("detection"), has("latency"), has("robustness"), has("calibration")]
    n_det = sum(1 for r in res if r.startswith("detection"))
    S_ = lambda label, state, hint="": {"label": label, "state": state, "hint": hint}
    return [
        S_("Detection, tracking, event log, export", "done"),
        S_("Abstention, temporal smoothing and optional calibration (off by default) in the app", "done"),
        S_("Perception Lab (simulated conditions)", "done"),
        S_("Sign detector trained on real sign data (GTSDB, German)", "done" if (names & gtsdb) or (len(names) not in (0, 80)) else "todo", "scripts\\phase5_gtsdb.ps1"),
        S_("Measured detection, latency, robustness, calibration", "done" if all(measured) else ("partial" if any(measured) else "todo"), "scripts\\evaluate_gtsdb.ps1"),
        S_("Model comparison (yolo11n with three seeds, and yolo11s)", "done" if n_det >= 2 else "todo", "phase5_gtsdb.ps1 -Name gtsdb_s -Base yolo11s.pt"),
        S_("Temporal smoothing study", "done" if has("temporal") else "todo", "python -m evaluation.temporal_eval"),
        S_("Indian-sign detector (skipped: no accessible labelled dataset)", "done" if len(indian) >= 5 else "todo", "scripts\\phase7_indian.ps1"),
    ]


@app.get("/api/sources")
def sources():
    """Never opens a camera: opening one can wake a phone / virtual camera (Phone Link) or hang. Use /api/cameras on demand."""
    vids = sorted(p.name for p in UPLOADS.iterdir() if p.suffix.lower() in VIDEO_EXT)
    return {"webcam_available": True, "videos": vids, "stream_active": S.stream_active}


def _probe_camera(i: int) -> dict:
    try:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if _is_windows() else cv2.VideoCapture(i)
        ok = bool(cap.isOpened()) and bool(cap.read()[0])
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        thumb = None
        if ok:                                      # small preview so you can SEE which index is the phone and which is the laptop webcam
            okf, fr = cap.read()
            if okf and fr is not None:
                sc = 200 / max(fr.shape[1], 1)
                small = cv2.resize(fr, (200, max(int(fr.shape[0] * sc), 1)))
                good, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                thumb = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if good else None
        cap.release()
        return {"index": i, "ok": ok, "width": w, "height": h, "thumb": thumb, "error": None if ok else "could not open or read a frame"}
    except Exception as e:
        return {"index": i, "ok": False, "width": 0, "height": 0, "thumb": None, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/cameras")
async def cameras(max_index: int = Query(4, ge=0, le=8)):
    """On-demand camera finder (opens each index briefly, with a timeout so a stuck virtual camera cannot hang the app)."""
    async def one(i: int) -> dict:
        try:
            return await asyncio.wait_for(asyncio.to_thread(_probe_camera, i), timeout=6)
        except asyncio.TimeoutError:
            return {"index": i, "ok": False, "width": 0, "height": 0, "thumb": None,
                    "error": "no response in 6 s (a phone or virtual camera that is still connecting?)"}
    return list(await asyncio.gather(*(one(i) for i in range(max_index + 1))))


def _is_windows() -> bool:
    import os
    return os.name == "nt"


def _render(frame: np.ndarray, dets: list[dict], source: str) -> bytes:
    """Draw + encode on a downscaled copy (cheaper than full-resolution), and log events."""
    h, w = frame.shape[:2]
    max_w = S.cfg["stream"]["max_width"]
    k = min(1.0, max_w / w)
    small = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
    ann = draw_detections(small, [dict(d, box=[v * k for v in d["box"]]) for d in dets]) if dets else small
    jpg = encode_jpeg(ann)
    S.events.update(dets, frame, jpg, source)
    return jpg


def _process(frame: np.ndarray, source: str, track: bool) -> dict:
    t0 = time.perf_counter()
    dets, infer_ms = S.engine.infer(frame, S.rs, track)
    jpg = _render(frame, dets, source)
    return {"dets": dets, "infer_ms": round(infer_ms, 1), "e2e_ms": round((time.perf_counter() - t0) * 1000, 1), "jpeg": jpg}


@app.post("/api/infer/image")
async def infer_image(file: UploadFile = File(...)):
    require_ready()
    img = decode_image(await read_upload(file, IMAGE_EXT))
    res = await asyncio.to_thread(_process, img, "image", False)
    return {"detections": res["dets"], "infer_ms": res["infer_ms"], "image_b64": base64.b64encode(res["jpeg"]).decode(),
            "disclaimer": DISCLAIMER}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    data = await read_upload(file, VIDEO_EXT)
    name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    (UPLOADS / name).write_bytes(data)
    cap = cv2.VideoCapture(str(UPLOADS / name))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        (UPLOADS / name).unlink(missing_ok=True)
        raise HTTPException(400, "OpenCV could not decode this video. Try H.264 .mp4.")
    return {"video": name}


@app.post("/api/stream/stop")
def stream_stop():
    S.stream_stop = True
    return {"stopping": S.stream_active}


@app.get("/api/telemetry")
def telemetry():
    return {"latency": S.engine.latency_stats(), "stream": S.stream_stats, "events_total": len(S.events.events)}


@app.get("/api/events")
def list_events(class_name: str | None = None, limit: int = Query(100, ge=1, le=500)):
    return S.events.list(class_name, limit)


@app.get("/api/events/export")
def export_events(format: str = Query("csv", pattern="^(csv|json)$")):
    body, mime = S.events.export(format)
    return Response(body, media_type=mime, headers={"Content-Disposition": f"attachment; filename=sanketra_events.{format}"})


@app.get("/api/events/{eid}/frame")
def event_frame(eid: int):
    jpg = S.events.frame(eid)
    if jpg is None:
        raise HTTPException(404, "No frame stored for this event (it may have been evicted).")
    return Response(jpg, media_type="image/jpeg")


@app.delete("/api/events")
def clear_events():
    S.events.clear()
    return {"cleared": True}


@app.post("/api/lab/compare")
async def lab_compare(file: UploadFile = File(...), corruption: str = Form("blur"), severity: int = Form(2)):
    """Run the same image clean and corrupted. Corruption is SIMULATED (see vision/corruptions.py)."""
    require_ready()
    if corruption not in CORRUPTIONS or not 0 <= severity <= 5:
        raise HTTPException(422, f"corruption must be in {CORRUPTIONS}, severity 0-5")
    img = decode_image(await read_upload(file, IMAGE_EXT))
    bad = apply_corruption(img, corruption, severity)
    a = await asyncio.to_thread(_process, img, "lab-clean", False)
    b = await asyncio.to_thread(_process, bad, f"lab-{corruption}{severity}", False)
    pack = lambda r: {"detections": r["dets"], "infer_ms": r["infer_ms"], "image_b64": base64.b64encode(r["jpeg"]).decode()}
    return {"clean": pack(a), "corrupted": pack(b), "corruption": corruption, "severity": severity, "simulated": True}


SAMPLES = ROOT / "data" / "samples"


@app.get("/api/samples")
def list_samples():
    return sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in IMAGE_EXT) if SAMPLES.exists() else []


@app.get("/api/samples/{name}")
def get_sample(name: str):
    p = SAMPLES / Path(name).name
    if not p.exists() or p.suffix.lower() not in IMAGE_EXT:
        raise HTTPException(404, "sample not found")
    return Response(p.read_bytes(), media_type="image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png")


@app.get("/api/results")
def results():
    """Measured experiment outputs only. Empty until you run the evaluation scripts."""
    out = {}
    for p in sorted((HOME / "experiments" / "results").glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            out[p.stem] = {"error": f"unreadable: {e}"}
    return out


@app.get("/api/results/{name}.png")
def result_plot(name: str):
    p = HOME / "experiments" / "results" / f"{Path(name).name}.png"
    if not p.exists():
        raise HTTPException(404, "plot not found")
    return Response(p.read_bytes(), media_type="image/png")


# ----------------------------- WebSocket --------------------------
def _open_webcam_checked(i: int):
    """Open camera i and require one real frame (a phone/virtual camera that is not ready fails here)."""
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if _is_windows() else cv2.VideoCapture(i)
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        raise IOError(f"camera {i} gave no frame")
    return cap


async def open_webcam_auto(preferred=None):
    """Try the last working camera, then 0..3; each gets 5 s so a stuck phone camera cannot hang the app."""
    state = HOME / "camera.json"
    order: list[int] = []
    if preferred is not None:
        order.append(preferred)
    else:
        try:
            order.append(int(json.loads(state.read_text(encoding="utf-8"))["index"]))
        except Exception:
            pass
    order += [i for i in range(4) if i not in order]
    tried = []
    for i in order:
        task = asyncio.ensure_future(asyncio.to_thread(_open_webcam_checked, i))
        try:
            cap = await asyncio.wait_for(asyncio.shield(task), timeout=5)
            try:
                state.write_text(json.dumps({"index": i}), encoding="utf-8")
            except OSError:
                pass
            return cap, i
        except asyncio.TimeoutError:
            tried.append(f"camera {i} (no response)")
            task.add_done_callback(lambda t: (t.exception() is None and t.result().release()) if not t.cancelled() else None)
        except Exception as e:
            tried.append(f"camera {i} ({e})")
    raise IOError("no working camera found. Tried: " + ", ".join(tried) +
                  ". Close other apps using the camera, disconnect a phone camera (Phone Link), then try again.")


def open_source(source: str, video: str | None, index: int):
    if source == "webcam":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if _is_windows() else cv2.VideoCapture(index)
    else:
        if not video:
            raise ValueError("video name required")
        path = UPLOADS / Path(video).name
        if not path.exists():
            raise FileNotFoundError(f"video '{video}' not found in data/uploads")
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise IOError("could not open source (camera busy/unplugged, or unsupported codec)")
    return cap


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket, source: str = "video", video: str | None = None, index: int = -1):
    await ws.accept()
    if S.engine.status != "ready":
        await ws.send_json({"type": "error", "message": f"Model not ready ({S.engine.status}): {S.engine.error}"})
        return await ws.close()
    if S.stream_active:
        await ws.send_json({"type": "error", "message": "Another stream is already active. Stop it first."})
        return await ws.close()
    cam_index = None
    try:
        if source == "webcam" and index < 0:
            forced = S.cfg.get("camera", {}).get("index", "auto")
            cap, cam_index = await open_webcam_auto(None if forced == "auto" else int(forced))
        else:
            cap = await asyncio.wait_for(asyncio.to_thread(open_source, source, video, max(index, 0)), timeout=10)
            cam_index = index if source == "webcam" else None
            if source == "webcam":                      # remember a camera the person chose by hand
                try:
                    (HOME / "camera.json").write_text(json.dumps({"index": index}), encoding="utf-8")
                except OSError:
                    pass
    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": f"Camera {index} did not respond within 10 s."})
        return await ws.close()
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        return await ws.close()
    S.stream_active, S.stream_stop = True, False
    S.engine.reset_tracking()
    S.events.new_session()
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx, ema_fps, last, fails = 0, 0.0, time.perf_counter(), 0
    seen_tracks: set = set()
    pending: asyncio.Task | None = None

    async def render_send(frame, dets, infer_ms, t_read, idx, fps):
        """Stage 2 (runs while the next frame is being read and inferred): draw, encode, log, send."""
        jpg = await asyncio.to_thread(_render, frame, dets, source)
        seen_tracks.update(d["track_id"] for d in dets if d["track_id"] is not None)
        st = S.engine.latency_stats()
        meta = json.dumps({"type": "frame", "frame": idx, "dets": dets, "telemetry": {
            "fps": round(fps, 1), "infer_ms": round(infer_ms, 1), "e2e_ms": round((time.perf_counter() - t_read) * 1000, 1),
            "src_fps": round(fps_src, 1) if source == "video" else None, "n_dets": len(dets), "unique_tracks": len(seen_tracks),
            "median_ms": st.get("median_ms"), "p95_ms": st.get("p95_ms")}}, separators=(",", ":")).encode()
        # binary message: 4-byte big-endian meta length | meta JSON | JPEG bytes (no base64, one atomic message)
        await ws.send_bytes(len(meta).to_bytes(4, "big") + meta + jpg)

    await ws.send_json({"type": "status", "message": f"{source} stream started" + (f" (camera {cam_index})" if cam_index is not None else "")})
    try:
        while not S.stream_stop:
            t_loop = time.perf_counter()
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                fails += 1
                if source == "video" and fails < 2:       # loop the clip forever; fails>=2 means rewind also failed
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    S.engine.reset_tracking(); S.events.new_session()   # IDs restart at the loop point
                    continue
                if source == "webcam" and fails <= 3:       # camera hiccup: try to reconnect
                    await ws.send_json({"type": "status", "message": f"camera read failed, reconnecting ({fails}/3)"})
                    cap.release(); await asyncio.sleep(1.0)
                    try:
                        cap = open_source(source, video, index)
                    except Exception:
                        pass
                    continue
                await ws.send_json({"type": "error", "message": "Source ended or camera disconnected."})
                break
            fails = 0
            dets, infer_ms = await asyncio.to_thread(S.engine.infer, frame, S.rs, True)   # stage 1 (GPU)
            if pending is not None:
                await pending                              # at most one frame in flight: bounded latency, no queue
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last, ema_fps = now, inst if ema_fps == 0 else 0.85 * ema_fps + 0.15 * inst
            frame_idx += 1
            S.stream_stats = {"fps": round(ema_fps, 1), "frame": frame_idx}
            job = render_send(frame, dets, infer_ms, t_loop, frame_idx, ema_fps)
            if S.cfg["stream"].get("overlap", False):      # optional: encode frame N while inferring frame N+1
                pending = asyncio.create_task(job)
            else:
                await job                                   # default: serial, the configuration measured fast on the target laptop
            if source == "video" and S.rs.pace:                # pace to source FPS (no fast-forward)
                await asyncio.sleep(max(0.0, 1.0 / fps_src - (time.perf_counter() - t_loop)))
    except (WebSocketDisconnect, RuntimeError):
        log.info("client disconnected")
    except Exception as e:
        log.exception("stream failed")
        try:
            await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
    finally:
        if pending is not None:
            try:
                await pending
            except Exception:
                pass
        cap.release()
        S.stream_active, S.stream_stats = False, {}
        try:
            await ws.close()
        except Exception:
            pass


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


_dist = ROOT / "frontend" / "dist"


class UIFiles(StaticFiles):
    """Serve the built UI. HTML is never cached (so a new build is picked up immediately);
    hashed assets are cached for a year."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        path = path.replace("\\", "/")              # Windows hands us backslashes ("assets\\index-xxx.js")
        if (resp.headers.get("content-type") or "").startswith("text/html"):
            resp.headers["Cache-Control"] = "no-store"
        elif path.startswith("assets/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp


def ui_build_id() -> str | None:
    """Name of the main JS bundle in the built UI (the browser compares it with the one it is running)."""
    try:
        m = re.search(r'/assets/index-[\w-]+\.js', (_dist / "index.html").read_text(encoding="utf-8"))
        return m.group(0) if m else None
    except OSError:
        return None


if _dist.exists():
    app.mount("/", UIFiles(directory=_dist, html=True), name="ui")
