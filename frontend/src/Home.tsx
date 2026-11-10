import { useEffect, useState } from "react";
import { api, Sign, StatusItem } from "./api";
import { useCount, useInView, useReveal, useScrollVar } from "./hooks";

type Go = (t: "console" | "lab" | "results" | "signs") => void;

function useLive() {
  const [m, setM] = useState<any>(null); const [t, setT] = useState<any>(null); const [err, setErr] = useState(false);
  useEffect(() => {
    let dead = false;
    const tick = () => Promise.all([api.model(), api.telemetry()]).then(([a, b]) => { if (!dead) { setM(a); setT(b); setErr(false); } }).catch(() => !dead && setErr(true));
    tick(); const id = setInterval(tick, 2500); return () => { dead = true; clearInterval(id); };
  }, []);
  return { m, t, err };
}

const Hl = ({ children }: { children: React.ReactNode }) => <span className="hl">{children}</span>;

const STEPS = [
  { k: "01", t: "Capture", d: "OpenCV reads a webcam or road video one frame at a time. Frames are never queued, so latency cannot silently pile up.", tag: "OpenCV · back-pressure" },
  { k: "02", t: "Detect", d: "A YOLO11 detector finds sign-like objects and draws real boxes. Half precision is used automatically on your NVIDIA GPU.", tag: "YOLO11 · CUDA fp16" },
  { k: "03", t: "Track", d: "ByteTrack keeps one identity per sign across frames, so a sign becomes one event with a first-seen and last-seen time.", tag: "ByteTrack · event log" },
  { k: "04", t: "Interpret", d: "Scores can be temperature-calibrated (off by default), are smoothed over time, and are flagged UNCERTAIN when evidence is weak. Meanings come from a sign registry.", tag: "Calibration · abstention" },
];

export default function Home({ go }: { go: Go }) {
  const { m, t, err } = useLive();
  const [signs, setSigns] = useState<Sign[]>([]);
  const [status, setStatus] = useState<StatusItem[]>([]);
  const tlRef = useScrollVar<HTMLDivElement>();
  const [statRef, statSeen] = useInView<HTMLDivElement>();
  useEffect(() => { api.registry().then(r => setSigns(r.signs)).catch(() => {}); api.status().then(setStatus).catch(() => {}); }, []);
  useReveal(signs.length);
  const cNum = useCount(m?.num_classes ?? 0, statSeen);
  const cReg = useCount(signs.length, statSeen);

  return (
    <div className="home">
      <section className="hero">
        <div className="reveal in eyebrow"><span className="live" />Educational prototype · traffic-sign perception</div>
        <h1 className="reveal in d1">Machines that read signs.<br /><Hl>Honest about what they don’t know.</Hl></h1>
        <p className="reveal in d2 lead">Sanketra detects and tracks traffic signs from a webcam or road video, explains what each detected sign group means, and flags its own uncertainty instead of pretending every prediction is correct.</p>
        <div className="reveal in d3 cta">
          <button className="btn primary lg" onClick={() => go("console")}>Open live console</button>
          <button className="btn lg" onClick={() => go("lab")}>Try the Perception Lab</button>
        </div>
        <div className="reveal in d4 chips">
          {err ? <span className="chip bad">Backend offline</span> : m ? (<>
            <span className="chip ok"><i className="live" />Model {m.status}</span><span className="chip">{m.name}</span><span className="chip">{m.device}</span>
            <span className="chip">{t?.latency?.median_ms != null ? `${t.latency.median_ms} ms median` : "no inference yet"}</span></>) : <span className="chip">connecting…</span>}
        </div>
        <div className="scrollhint" aria-hidden><span /></div>
      </section>

      <section className="sect">
        <div className="reveal head"><div className="kicker">How it works</div><h2>From pixels to a <Hl>trustworthy event</Hl></h2>
          <p className="dim">Every stage below is implemented and running in this app. Nothing on this page is a mock-up.</p></div>
        <div className="timeline" ref={tlRef}>
          <div className="tl-line"><div /></div>
          {STEPS.map((s, i) => (
            <div key={s.k} className={`tl-item reveal ${i % 2 ? "right" : "left"}`}>
              <div className="tl-dot" /><div className="tl-card"><div className="mono dim small">{s.k}</div><h3>{s.t}</h3><p>{s.d}</p><span className="chip">{s.tag}</span></div>
            </div>))}
        </div>
      </section>

      <section className="sect">
        <div className="reveal head"><div className="kicker">Honest by design</div><h2>Confidence is <Hl>earned, not assumed</Hl></h2></div>
        <div className="cards3">
          <div className="reveal card fx1"><div className="ill ill-cal" aria-hidden><div className="bar"><div /></div><div className="bar"><div /></div><div className="bar"><div /></div></div>
            <h3>Calibration</h3><p>Raw detector scores are often mis-calibrated (for these German-data models they were under-confident). Temperature scaling, fitted on validation data, brought test ECE from 0.122 to 0.070 for one model; it is a slider in the Console and off by default.</p><span className="dim small">Illustration</span></div>
          <div className="reveal card d1"><div className="ill ill-abs" aria-hidden><div className="box" /></div>
            <h3>Abstention</h3><p>Below your chosen threshold, a detection turns into a dashed amber UNCERTAIN box. Weak evidence is shown as weak.</p><span className="dim small">Illustration</span></div>
          <div className="reveal card d2"><div className="ill ill-rob" aria-hidden><i /><i /><i /><i /><i /></div>
            <h3>Robustness lab</h3><p>Apply simulated low light, blur, noise and low resolution, then see how detections change. Dataset-level curves appear after you run the experiments.</p><span className="dim small">Illustration</span></div>
        </div>
      </section>

      <section className="sect">
        <div className="reveal head"><div className="kicker">Live from your machine</div><h2>The numbers here are <Hl>real</Hl></h2></div>
        <div className="statgrid reveal" ref={statRef}>
          <div className="statbig"><span>Model</span><b>{m?.name ?? "—"}</b></div>
          <div className="statbig"><span>Device</span><b>{m?.device ?? "—"}</b></div>
          <div className="statbig"><span>Parameters</span><b>{m ? `${m.params_million} M` : "—"}</b></div>
          <div className="statbig"><span>Model classes</span><b className="mono">{m ? cNum : "—"}</b></div>
          <div className="statbig"><span>Registry entries</span><b className="mono">{cReg}</b></div>
          <div className="statbig"><span>Events logged</span><b className="mono">{t?.events_total ?? "—"}</b></div>
        </div>
        {m?.is_coco_demo_model && <p className="warnbox reveal" style={{ maxWidth: 720, margin: "18px auto 0" }}>This is the general-purpose COCO demo model. It knows “stop sign” only. Train on sign data to cover Indian signs.</p>}
      </section>

      <section className="sect">
        <div className="reveal head"><div className="kicker">Sign registry</div><h2>A versioned record of <Hl>what each sign means</Hl></h2>
          <p className="dim">Each entry says whether its wording was checked and against which source; unverified entries are drafts.</p></div>
        <div className="signrow">
          {signs.slice(0, 6).map((s, i) => (<div key={s.class_name} className={`reveal card sm d${i % 3}`}><span className={`cat cat-${s.category}`}>{s.category}</span><h3>{s.label_en}</h3><p className="dim small">{s.meaning}</p><span className="chip warn">{s.verified ? "verified" : "unverified draft"}</span></div>))}
        </div>
        <div className="reveal center"><button className="btn" onClick={() => go("signs")}>Browse the registry</button></div>
      </section>

      <section className="sect">
        <div className="reveal head"><div className="kicker">Status</div><h2>What is done, and <Hl>what is not</Hl></h2></div>
        <div className="reveal table">
          {status.map(it => (<div className="trow" key={it.label}><span>{it.label}{it.hint && it.state === "todo" && <span className="dim small mono"> · {it.hint}</span>}</span>
            <span className={`chip ${it.state === "done" ? "ok" : it.state === "planned" ? "dim" : "warn"}`}>{it.state === "done" ? "Done" : it.state === "partial" ? "Partly done" : it.state === "planned" ? "Planned" : "Not done"}</span></div>))}
          {status.length === 0 && <div className="trow dim">Loading status…</div>}
        </div>
      </section>

      <section className="sect final reveal"><h2>Ready when you are.</h2><button className="btn primary lg" onClick={() => go("console")}>Open live console</button>
        <p className="dim small" style={{ marginTop: 18 }}>Educational prototype. Not a certified automotive safety system. Detections can be wrong, even at high confidence.</p></section>
    </div>
  );
}
