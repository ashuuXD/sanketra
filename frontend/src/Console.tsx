import { memo, useCallback, useEffect, useRef, useState } from "react";
import { api, Cam, Det, Ev, Settings, Sign, Telemetry } from "./api";

const catLabel = (c: string) => (c === "unknown" ? "not a registered sign" : c);
const fmt = (n: number | null | undefined, d = 1) => (n == null ? "—" : n.toFixed(d));
const Stat = ({ k, v }: { k: string; v: string }) => <div className="stat"><span>{k}</span><b className="mono">{v}</b></div>;
const Slider = ({ label, v, min, max, step, on }: { label: string; v: number; min: number; max: number; step: number; on: (v: number) => void }) => (
  <label className="sl"><span>{label}</span><b className="mono">{v.toFixed(2)}</b><input type="range" min={min} max={max} step={step} value={v} onChange={e => on(+e.target.value)} /></label>);

export default function Console() {
  const [hasFrame, setHasFrame] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pendingBlob = useRef<Blob | null>(null);
  const decoding = useRef(false);
  const latest = useRef<{ dets: Det[]; tel: Telemetry } | null>(null);
  const [dets, setDets] = useState<Det[]>([]);
  const [tel, setTel] = useState<Telemetry | null>(null);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState("");
  const [src, setSrc] = useState<{ videos: string[] }>({ videos: [] });
  const [video, setVideo] = useState("");
  const [model, setModel] = useState<any>(null);
  const [reg, setReg] = useState<Record<string, Sign>>({});
  const [set_, setSet] = useState<Settings | null>(null);
  const [srcName, setSrcName] = useState("");
  const [cams, setCams] = useState<Cam[] | null>(null);
  const [finding, setFinding] = useState(false);
  const [cam, setCam] = useState<number>(() => { const v = Number(localStorage.getItem("sanketra.cam")); return Number.isFinite(v) && localStorage.getItem("sanketra.cam") !== null ? v : -1; });
  const pickCam = (i: number) => { setCam(i); localStorage.setItem("sanketra.cam", String(i)); };
  const findCams = async () => { setFinding(true); setErr(""); try { setCams(await api.cameras()); } catch (x) { setErr(String(x)); } setFinding(false); };
  const ws = useRef<WebSocket | null>(null);
  const running = status === "streaming" || status === "reconnecting";

  const refreshSources = useCallback(() => api.sources().then(s => { setSrc(s); setVideo(v => v || s.videos[0] || ""); }).catch(e => setErr(String(e))), []);
  useEffect(() => { refreshSources(); api.model().then(setModel).catch(() => {}); api.getSettings().then(setSet).catch(() => {}); api.registry().then(r => setReg(Object.fromEntries(r.signs.map(s => [s.class_name, s])))).catch(() => {}); }, [refreshSources]);
  // UI state is refreshed at ~5 Hz; frames are painted straight to a canvas (no React render per frame)
  useEffect(() => { const t = setInterval(() => { const l = latest.current; if (l) { latest.current = null; setDets(l.dets); setTel(l.tel); } }, 200); return () => clearInterval(t); }, []);
  const paint = useCallback(async () => {
    if (decoding.current) return; decoding.current = true;
    try {
      while (pendingBlob.current) {            // if frames arrive faster than we can decode, only the newest is drawn
        const b = pendingBlob.current; pendingBlob.current = null;
        const bmp = await createImageBitmap(b); const cv = canvasRef.current;
        if (cv) { if (cv.width !== bmp.width || cv.height !== bmp.height) { cv.width = bmp.width; cv.height = bmp.height; } cv.getContext("2d")!.drawImage(bmp, 0, 0); }
        bmp.close();
      }
    } finally { decoding.current = false; }
  }, []);
  useEffect(() => () => { ws.current?.close(); }, []);

  const start = (source: "webcam" | "video") => {
    setErr(""); setStatus("connecting"); setSrcName(source === "webcam" ? "Webcam" : video);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const q = source === "webcam" ? `source=webcam&index=${cam}` : `source=video&video=${encodeURIComponent(video)}`;
    const s = new WebSocket(`${proto}://${location.host}/ws/stream?${q}`); ws.current = s;
    s.binaryType = "arraybuffer";
    s.onmessage = e => {
      if (typeof e.data !== "string") {         // binary frame: 4-byte meta length | meta JSON | JPEG
        const buf = e.data as ArrayBuffer; const n = new DataView(buf).getUint32(0);
        const m = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
        pendingBlob.current = new Blob([new Uint8Array(buf, 4 + n)], { type: "image/jpeg" }); paint();
        latest.current = { dets: m.dets, tel: m.telemetry };
        setStatus(p => (p === "streaming" ? p : "streaming")); setHasFrame(true);
        return;
      }
      const m = JSON.parse(e.data);
      if (m.type === "error") { setErr(m.message); setStatus("error"); }
      else if (m.type === "status") { const cm = /camera (\d+)/.exec(m.message); if (cm) setSrcName(`Webcam (camera ${cm[1]})`); setStatus(m.message.includes("started") ? "streaming" : "reconnecting"); }
    };
    s.onclose = () => setStatus(p => (p === "error" ? p : "idle")); s.onerror = () => { setErr("WebSocket connection failed"); setStatus("error"); };
  };
  const stop = () => { api.stop(); ws.current?.close(); setStatus("idle"); setDets([]); };
  const [autoCam, setAutoCam] = useState<boolean>(() => localStorage.getItem("sanketra.autocam") === "1");
  const toggleAuto = (v: boolean) => { setAutoCam(v); localStorage.setItem("sanketra.autocam", v ? "1" : "0"); };
  useEffect(() => { if (autoCam && cam >= 0) start("webcam"); }, []); // opt-in, and only for a camera you chose yourself; otherwise nothing starts
  const upd = (p: Partial<Settings>) => { setSet(s => (s ? { ...s, ...p } : s)); api.settings(p).catch(e => setErr(String(e))); };

  const signDets = dets.filter(d => d.category !== "object");          // traffic-sign detections take priority in the panel
  const top = [...(signDets.length ? signDets : dets)].sort((a, b) => b.conf - a.conf)[0];
  const info = top && top.category !== "object" ? reg[top.class_name] : undefined;

  return (<>
    <div className="console">
      <section className="panel view">
        <div className="bar">
          <span className={`pill ${status}`}>{status}</span>
          <select value={video} onChange={e => setVideo(e.target.value)} disabled={running}>{src.videos.length === 0 && <option value="">no videos yet</option>}{src.videos.map(v => <option key={v}>{v}</option>)}</select>
          <label className="btn">Upload video<input hidden type="file" accept="video/*" onChange={async e => { const f = e.target.files?.[0]; if (!f) return; try { const r = await api.upload("/api/video/upload", f); await refreshSources(); setVideo(r.video); } catch (x) { setErr(String(x)); } e.target.value = ""; }} /></label>
          {!running ? (<><button className="btn primary" disabled={!video} onClick={() => start("video")}>Play video</button>
            <button className="btn" onClick={() => start("webcam")}>Webcam{cam >= 0 ? ` (camera ${cam})` : ""}</button>
            <button className="btn" onClick={findCams} disabled={finding}>{finding ? "Finding…" : "Find cameras"}</button>
            <label className="small dim" style={{ display: "flex", gap: 6, alignItems: "center" }} title={cam < 0 ? "Choose a camera with Find cameras first" : "Start the chosen camera when this page opens"}>
              <input type="checkbox" checked={autoCam} disabled={cam < 0} onChange={e => toggleAuto(e.target.checked)} /> Auto-start camera</label></>)
            : <button className="btn warn" onClick={stop}>Stop</button>}
        </div>
        {cams && !running && (<div className="camfind" style={{ display: "flex", gap: 10, flexWrap: "wrap", padding: "8px 4px" }}>
          {cams.map(c => (<button key={c.index} className="btn" disabled={!c.ok} onClick={() => pickCam(c.index)} title={c.error ?? ""}
            style={{ padding: 6, display: "flex", flexDirection: "column", gap: 4, alignItems: "center", outline: cam === c.index ? "2px solid var(--accent, #2dd4bf)" : "none" }}>
            {c.thumb ? <img src={c.thumb} alt={`camera ${c.index}`} style={{ width: 120, borderRadius: 4 }} /> : <span style={{ width: 120, height: 68, display: "grid", placeItems: "center", opacity: .6 }}>no picture</span>}
            <span className="mono" style={{ fontSize: 12 }}>camera {c.index}{c.ok ? ` · ${c.width}×${c.height}` : " · unavailable"}{cam === c.index ? " · chosen" : ""}</span></button>))}
          <button className="btn" onClick={() => pickCam(-1)} style={{ alignSelf: "center" }}>{cam === -1 ? "Automatic · chosen" : "Use automatic"}</button>
        </div>)}
        <div className={`stage ${running ? "live" : ""}`}>
          <i className="cr tl" /><i className="cr tr" /><i className="cr bl" /><i className="cr br" />
          <canvas ref={canvasRef} className="vid" style={{ display: hasFrame ? "block" : "none" }} />
          {!hasFrame && <div className="empty">{status === "connecting" ? "Connecting…" : running ? "Starting stream… the first frame can take a few seconds." : "Pick a video and press Play, or press Find cameras, click the one that shows YOU (not a phone), then Webcam."}</div>}
          {running && <div className="hud mono">{srcName} · {fmt(tel?.fps)} FPS</div>}
          {err && <div className="err">{err}</div>}
        </div>
        <div className="telemetry">
          <Stat k="FPS" v={fmt(tel?.fps)} /><Stat k="Source FPS" v={tel?.src_fps ? fmt(tel.src_fps, 0) : "—"} /><Stat k="Inference" v={`${fmt(tel?.infer_ms)} ms`} />
          <Stat k="End-to-end" v={`${fmt(tel?.e2e_ms)} ms`} /><Stat k="Median / p95" v={`${fmt(tel?.median_ms)} / ${fmt(tel?.p95_ms)}`} />
          <Stat k="In frame" v={String(tel?.n_dets ?? "—")} /><Stat k="Unique tracks" v={String(tel?.unique_tracks ?? "—")} />
        </div>
      </section>

      <aside className="panel side">
        <h3>Detection intelligence</h3>
        <div className="intel">
          {top ? (<>
            <div className="big">{top.label_en}</div>
            <div className="row"><span className={`cat cat-${top.category}`}>{catLabel(top.category)}</span><span className="mono">{(top.conf * 100).toFixed(1)}%</span>{top.uncertain && <span className="cat cat-uncertain">uncertain</span>}</div>
            <div className="meter"><div style={{ width: `${top.conf * 100}%` }} /></div>
            <p className="dim mono small">raw {top.raw_conf.toFixed(3)} → shown {top.conf.toFixed(3)}</p>
            {info ? <><p>{info.meaning}</p><p className="dim small">{info.guidance}</p><p className="dim small">{info.verified ? "Verified: " : "Unverified: "}{info.source ?? "no source"}</p></>
              : top.category === "object" ? <p className="dim small">General object seen by the second (COCO) detector. It is not a traffic sign.</p>
              : <p className="dim small">No registry entry for “{top.class_name}”. Shown as an unverified detection.</p>}
          </>) : <p className="dim">No detection in the current frame.</p>}
        </div>
        <p className="dim small">Confidence is a model score, not a guarantee. Prototype only.</p>
        <h3>Model diagnostics</h3>
        {model && <ul className="kv"><li><span>Model</span>{model.name}</li><li><span>Device</span>{model.device}</li><li><span>Size</span>{model.size_mb} MB · {model.params_million}M</li><li><span>Classes</span>{model.num_classes}</li></ul>}
        {model?.training_note && <p className="warnbox">{model.training_note}</p>}
        {model?.is_coco_demo_model && <p className="warnbox">COCO demo model: only “stop sign” is a traffic sign it knows. Train on sign data for real coverage.</p>}
        {set_ && <div className="ctl">
          <Slider label="Confidence threshold" v={set_.conf} min={0.05} max={0.95} step={0.05} on={v => upd({ conf: v })} />
          <Slider label="Abstain below" v={set_.abstain_below} min={0} max={1} step={0.05} on={v => upd({ abstain_below: v })} />
          <Slider label="Temperature (calibration)" v={set_.temperature} min={0.5} max={3} step={0.1} on={v => upd({ temperature: v })} />
          <label className="sel">Inference size <select value={set_.imgsz} onChange={e => upd({ imgsz: +e.target.value })}>{[320, 416, 640].map(s => <option key={s}>{s}</option>)}</select></label>
          {model?.objects_available && <label className="chk"><input type="checkbox" checked={set_.objects} onChange={e => upd({ objects: e.target.checked })} /> Also detect people and vehicles (second model)</label>}
          <label className="chk"><input type="checkbox" checked={set_.tracking} onChange={e => upd({ tracking: e.target.checked })} /> Tracking (ByteTrack)</label>
          <label className="chk"><input type="checkbox" checked={set_.smoothing} onChange={e => upd({ smoothing: e.target.checked })} /> Temporal smoothing (EMA)</label>
          <label className="chk"><input type="checkbox" checked={set_.pace} onChange={e => upd({ pace: e.target.checked })} /> Play at source FPS (off = max speed)</label>
          <p className="dim small">FPS = whole-pipeline frames per second. Inference ms = model time only. With pacing on, FPS cannot exceed the video’s own frame rate. Accuracy is measured offline (see Results).</p>
        </div>}
      </aside>
    </div>

    <Timeline />
  </>);
}

const Timeline = memo(function Timeline() {
  const [events, setEvents] = useState<Ev[]>([]);
  const [filter, setFilter] = useState("");
  const [sel, setSel] = useState<Ev | null>(null);
  const [signsOnly, setSignsOnly] = useState(false);
  const last = useRef("");
  useEffect(() => {
    const tick = () => { if (document.hidden) return; api.events(filter).then(e => { const sig = e.map(x => `${x.id}:${x.hits}`).join(); if (sig !== last.current) { last.current = sig; setEvents(e); } }).catch(() => {}); };
    tick(); const t = setInterval(tick, 1500); return () => clearInterval(t);
  }, [filter]);
  const classes = Array.from(new Set(events.map(e => e.class_name)));
  return (<section className="panel feed">
    <div className="bar"><h3>Event timeline</h3>
      <select value={filter} onChange={e => setFilter(e.target.value)}><option value="">all classes</option>{classes.map(c => <option key={c}>{c}</option>)}</select>
      <label className="chk small"><input type="checkbox" checked={signsOnly} onChange={e => setSignsOnly(e.target.checked)} /> Signs only</label>
      <a className="btn" href="/api/events/export?format=csv">CSV</a><a className="btn" href="/api/events/export?format=json">JSON</a>
      <button className="btn" onClick={() => api.clear().then(() => { setEvents([]); setSel(null); last.current = ""; })}>Clear</button></div>
    <div className="evwrap"><div className="evlist">
      {events.length === 0 && <p className="dim pad">No events yet. An event is created when a sign is first detected.</p>}
      {events.filter(e => !signsOnly || e.category !== "object").map(e => (<div key={e.id} className={`ev ${sel?.id === e.id ? "on" : ""}`} onClick={() => setSel(e)}>
        {e.thumb ? <img alt="" src={`data:image/jpeg;base64,${e.thumb}`} /> : <div className="th" />}
        <div><b>{e.label_en}</b> <span className={`cat cat-${e.category}`}>{catLabel(e.category)}</span>
          <div className="dim small mono">{e.first_seen.slice(11, 23)} → {e.last_seen.slice(11, 23)} · {e.hits} hits · best {e.best_conf.toFixed(2)}{e.track_id != null ? ` · #${e.track_id % 100000}` : ""} · {e.source}</div></div></div>))}
    </div>
      <div className="evview">{sel ? <img alt="event frame" src={`/api/events/${sel.id}/frame`} /> : <p className="dim pad">Select an event to inspect its frame.</p>}</div></div>
  </section>);
});
