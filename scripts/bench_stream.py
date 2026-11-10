"""Measure end-to-end streaming throughput over the real WebSocket (server must be running).
  python scripts/bench_stream.py --port 8000 --seconds 12
Reports frames/s received, frame-gap median/p95/max and bytes per frame, with pacing OFF (max speed)."""
import argparse, asyncio, json, time, urllib.request
import numpy as np
import websockets

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--seconds", type=int, default=12)
ap.add_argument("--video", default="demo_pan.mp4"); a = ap.parse_args()


def post(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req).read()


async def main():
    for _ in range(120):                       # wait until the model is ready
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/api/health").read())["model_status"] == "ready":
                break
        except Exception:
            pass
        await asyncio.sleep(1)
    post("/api/settings", {"pace": False})
    ts, sizes = [], []
    async with websockets.connect(f"ws://127.0.0.1:{a.port}/ws/stream?source=video&video={a.video}", max_size=2**25) as ws:
        t_end = None
        while True:
            m = await ws.recv()
            if isinstance(m, str) and '"error"' in m[:30]:
                raise SystemExit(m)
            if isinstance(m, bytes) or (isinstance(m, str) and '"frame"' in m[:40]):
                now = time.perf_counter(); t_end = t_end or now + a.seconds
                ts.append(now); sizes.append(len(m))
                if now > t_end: break
    post("/api/stream/stop"); post("/api/settings", {"pace": True})
    ts = np.array(ts[3:]); gaps = np.diff(ts) * 1000
    print(f"frames={len(ts)} fps={len(ts) / (ts[-1] - ts[0]):.1f} gap_ms median={np.median(gaps):.0f} p95={np.percentile(gaps, 95):.0f} max={gaps.max():.0f} bytes/frame={int(np.mean(sizes))}")

asyncio.run(main())
