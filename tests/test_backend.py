import json
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.events import EventLog
from backend.registry import SignRegistry
from backend.settings import RuntimeSettings


def det(tid, name="stop sign", conf=0.9):
    return {"track_id": tid, "class_name": name, "label_en": "Stop", "category": "regulatory", "conf": conf, "uncertain": False, "box": [5, 5, 40, 40]}


def test_registry_known_and_unknown():
    r = SignRegistry()
    st = r.lookup("stop sign")
    assert st["category"] == "regulatory" and st["verified"] is True and "delhipolice" in st["source"] and st["verified_on"] == "2026-11-01"
    assert r.lookup("sharp_curve")["verified"] is False          # no exact official match: must stay unverified
    assert r.lookup("prohibitory")["verified"] is False          # GTSDB group entries are not Indian-sign verifications
    u = r.lookup("flying_saucer")
    assert u["category"] == "unknown" and u["label_en"] == "Flying Saucer"


def test_event_log_bounded_dedup_and_export():
    log, frame = EventLog(max_events=5), np.zeros((50, 50, 3), np.uint8)
    log.new_session()
    for _ in range(3):
        log.update([det(1)], frame, b"jpg", "video")
    assert len(log.list()) == 1 and log.list()[0]["hits"] == 3     # same track -> one event
    for t in range(2, 12):
        log.update([det(t)], frame, b"jpg", "video")
    assert len(log.events) == 5 and len(log.frames) == 5 and len(log.by_track) == 5   # bounded, no leaks
    assert log.export("csv")[0].splitlines()[0].startswith("id,session")
    assert isinstance(json.loads(log.export("json")[0]), list)


def test_runtime_settings_validation():
    with pytest.raises(ValueError):
        RuntimeSettings(imgsz=500)
    with pytest.raises(ValueError):
        RuntimeSettings(conf=2)


@pytest.fixture(scope="module")
def client():
    from backend.main import app, S
    with TestClient(app) as c:
        for _ in range(180):
            if S.engine.status in ("ready", "error"):
                break
            time.sleep(1)
        assert S.engine.status == "ready", S.engine.error
        yield c


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_health_model_registry(client):
    h = client.get("/api/health").json()
    assert h["model_status"] == "ready" and "Educational prototype" in h["disclaimer"]
    m = client.get("/api/model").json()
    assert m["device"] and m["params_million"] > 0 and m["num_classes"] > 0
    assert client.get("/api/registry").json()["signs"]


def test_settings_roundtrip_and_validation(client):
    assert client.post("/api/settings", json={"conf": 0.3}).json()["conf"] == 0.3
    assert client.post("/api/settings", json={"imgsz": 123}).status_code == 422
    assert client.post("/api/settings", json={"conf": 0.25}).status_code == 200


def test_infer_image_valid_and_invalid(client):
    p = Path(__file__).parent / "assets" / "street.jpg"
    r = client.post("/api/infer/image", files={"file": ("street.jpg", p.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["infer_ms"] > 0 and body["image_b64"] and len(body["detections"]) >= 1
    assert all(0 <= d["conf"] <= 1 for d in body["detections"])
    assert client.post("/api/infer/image", files={"file": ("x.jpg", b"garbage", "image/jpeg")}).status_code == 400
    assert client.post("/api/infer/image", files={"file": ("x.exe", b"x", "application/octet-stream")}).status_code == 415


def test_events_and_export_after_inference(client):
    ev = client.get("/api/events").json()
    assert ev and ev[0]["thumb"]
    assert client.get(f"/api/events/{ev[0]['id']}/frame").headers["content-type"] == "image/jpeg"
    assert client.get("/api/events/999999/frame").status_code == 404
    assert "label_en" in client.get("/api/events/export?format=csv").text
    assert client.get("/api/events/export?format=xml").status_code == 422


def test_lab_compare(client):
    p = Path(__file__).parent / "assets" / "street.jpg"
    r = client.post("/api/lab/compare", files={"file": ("s.jpg", p.read_bytes(), "image/jpeg")}, data={"corruption": "lowlight", "severity": "4"})
    assert r.status_code == 200 and r.json()["simulated"] is True
    assert client.post("/api/lab/compare", files={"file": ("s.jpg", p.read_bytes(), "image/jpeg")}, data={"corruption": "fog", "severity": "1"}).status_code == 422


def test_results_empty_or_measured(client):
    assert isinstance(client.get("/api/results").json(), dict)


def recv(ws):
    """Parse a stream message: JSON text (status/error) or binary frame = len(4) | meta JSON | JPEG."""
    m = ws.receive()
    if m.get("bytes") is None:
        return json.loads(m["text"])
    b = m["bytes"]
    n = int.from_bytes(b[:4], "big")
    meta = json.loads(b[4:4 + n]); meta["jpeg"] = b[4 + n:]
    return meta


def test_video_upload_and_websocket_stream(client, tmp_path):
    p = Path(__file__).parent / "assets" / "street.jpg"
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    vp = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(20):
        vw.write(np.roll(img, i * 3, axis=1))
    vw.release()
    up = client.post("/api/video/upload", files={"file": ("clip.mp4", vp.read_bytes(), "video/mp4")})
    assert up.status_code == 200
    name = up.json()["video"]
    with client.websocket_connect(f"/ws/stream?source=video&video={name}") as ws:
        frames = []
        while len(frames) < 4:
            m = recv(ws)
            assert m["type"] in ("status", "frame"), m
            if m["type"] == "frame":
                frames.append(m)
        t = frames[-1]["telemetry"]
        assert t["fps"] > 0 and t["infer_ms"] > 0 and t["e2e_ms"] > 0
        assert frames[-1]["jpeg"][:2] == b"\xff\xd8"          # real JPEG bytes, no base64
        client.post("/api/stream/stop")
    time.sleep(1.5)
    assert client.get("/api/sources").json()["stream_active"] is False   # resources released
    from backend.main import UPLOADS
    (UPLOADS / name).unlink(missing_ok=True)


def test_ws_missing_video_reports_error(client):
    with client.websocket_connect("/ws/stream?source=video&video=nope.mp4") as ws:
        assert recv(ws)["type"] == "error"


def test_untracked_detections_merge_instead_of_spamming():
    log, frame = EventLog(max_events=50), np.zeros((50, 50, 3), np.uint8)
    log.new_session()
    for _ in range(10):                                   # tracking off: same box every frame
        log.update([det(None)], frame, b"jpg", "video")
    assert len(log.events) == 1 and log.list()[0]["hits"] == 10
    log.update([det(None)], frame, b"jpg", "image"); log.update([det(None)], frame, b"jpg", "image")
    assert len(log.events) == 3                           # still images are always separate events


def test_pace_setting_and_samples(client):
    assert client.post("/api/settings", json={"pace": False}).json()["pace"] is False
    client.post("/api/settings", json={"pace": True})
    names = client.get("/api/samples").json()
    assert names and client.get(f"/api/samples/{names[0]}").status_code == 200
    assert client.get("/api/samples/..%2Fsecret.txt").status_code in (404, 422)


def test_ui_is_not_cached_and_build_id_reported(client):
    r = client.get("/")
    if r.status_code == 200:                                   # UI is built in this checkout
        assert r.headers["cache-control"] == "no-store"
        assert client.get("/api/health").json()["ui_build"].startswith("/assets/index-")
        asset = client.get("/api/health").json()["ui_build"]
        assert "immutable" in client.get(asset).headers["cache-control"]


def test_sources_never_probe_camera_and_finder_is_on_demand(client):
    s = client.get("/api/sources").json()
    assert s["webcam_available"] is True and "videos" in s          # no camera is opened just by loading the page
    cams = client.get("/api/cameras?max_index=1").json()
    assert [c["index"] for c in cams] == [0, 1] and all("ok" in c and "error" in c for c in cams)


def test_webcam_stream_reports_error_instead_of_hanging(client, monkeypatch):
    import backend.main as bm
    def no_camera(i):                                   # the test must not depend on whether THIS computer has a camera
        raise IOError(f"camera {i} gave no frame")
    monkeypatch.setattr(bm, "_open_webcam_checked", no_camera)
    with client.websocket_connect("/ws/stream?source=webcam") as ws:          # automatic camera choice
        m = recv(ws)
        assert m["type"] == "error" and "no working camera found" in m["message"] and "Tried" in m["message"]


def test_objects_setting_and_aux_conversion(client):
    from backend.main import S
    assert client.post("/api/settings", json={"objects": False}).json()["objects"] is False
    client.post("/api/settings", json={"objects": True})
    import torch

    class Boxes:
        xyxy = torch.tensor([[1., 2., 30., 40.]]); conf = torch.tensor([0.8]); cls = torch.tensor([0.]); id = torch.tensor([3.])
        def __len__(self): return 1

    class Res:
        boxes = Boxes(); names = {0: "person"}

    d = S.engine._to_dets(Res(), S.rs, aux=True)[0]
    assert d["category"] == "object" and d["label_en"] == "Person" and d["track_id"] == 100003
    assert d["track_id"] % 100000 == 3                               # the UI/overlay shows the original id


def test_status_and_registry_flags(client):
    st = client.get("/api/status").json()
    assert {i["state"] for i in st} <= {"done", "partial", "todo", "planned"} and len(st) >= 8
    assert next(i for i in st if i["label"].startswith("Indian-sign"))["state"] == "todo"        # honest: not trained here
    reg = client.get("/api/registry").json()
    assert all("in_model" in s for s in reg["signs"]) and "model_classes" in reg
