import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(autouse=True)
def _no_real_cameras(monkeypatch):
    """Unit tests must never open real camera hardware: a hung camera driver (Phone Link, a camera held by another
    program, right after a forced shutdown) leaves a blocked worker thread, and the test run then waits up to
    300 seconds per event loop for it at shutdown."""
    try:
        import backend.main as bm
    except Exception:          # pure-vision tests do not need the web app
        return
    monkeypatch.setattr(bm, "_probe_camera", lambda i: {"index": i, "ok": False, "width": 0, "height": 0, "thumb": None, "error": "no camera in tests"})

    def _no_camera(i):
        raise IOError(f"camera {i} is not available in tests")

    monkeypatch.setattr(bm, "_open_webcam_checked", _no_camera)
