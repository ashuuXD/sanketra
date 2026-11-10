import { useCallback, useEffect, useRef, useState } from "react";
import { api, Pack } from "./api";

const CORR: [string, string][] = [["lowlight", "Low light"], ["brightness", "Overexposure"], ["blur", "Motion blur"], ["noise", "Sensor noise"], ["lowres", "Low resolution"]];

const mean = (p: Pack) => (p.detections.length ? p.detections.reduce((a, d) => a + d.conf, 0) / p.detections.length : null);

export default function Lab() {
  const [blob, setBlob] = useState<Blob | null>(null);
  const [name, setName] = useState(""); const [preview, setPreview] = useState<string | null>(null);
  const [samples, setSamples] = useState<string[]>([]);
  const [corr, setCorr] = useState("lowlight"); const [sev, setSev] = useState(3); const [auto, setAuto] = useState(true);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState(""); const [drag, setDrag] = useState(false);
  const [res, setRes] = useState<{ clean: Pack; corrupted: Pack } | null>(null);
  const seq = useRef(0);

  useEffect(() => { api.samples().then(setSamples).catch(() => {}); }, []);
  const pick = (b: Blob, n: string) => { setBlob(b); setName(n); setRes(null); setErr(""); setPreview(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(b); }); };
  const loadSample = async (n: string) => { try { pick(await (await fetch(`/api/samples/${encodeURIComponent(n)}`)).blob(), n); } catch (e) { setErr(String(e)); } };

  const run = useCallback(async () => {
    if (!blob) return; const my = ++seq.current; setBusy(true); setErr("");
    try { const r = await api.upload("/api/lab/compare", blob, { corruption: corr, severity: String(sev) }, name || "image.jpg"); if (my === seq.current) setRes(r); }
    catch (e) { if (my === seq.current) setErr(String(e)); } finally { if (my === seq.current) setBusy(false); }
  }, [blob, corr, sev, name]);
  useEffect(() => { if (!auto || !blob) return; const t = setTimeout(run, 350); return () => clearTimeout(t); }, [auto, blob, corr, sev, run]);

  const col = (title: string, p: Pack | null, sub: string) => (<div className="panel labcol"><div className="lh"><h3>{title}</h3><span className="dim small">{sub}</span></div>
    <div className="labimg">{p ? <img alt={title} src={`data:image/jpeg;base64,${p.image_b64}`} /> : preview ? <img alt="preview" className="ghost" src={preview} /> : <div className="empty">Choose an image to begin</div>}</div>
    {p && <><p className="mono small">{p.detections.length} detection(s) · {p.infer_ms} ms</p>
      <ul className="kv">{p.detections.slice(0, 6).map((d, i) => <li key={i}><span>{d.label_en}</span>{(d.conf * 100).toFixed(1)}%{d.uncertain ? " · uncertain" : ""}</li>)}</ul></>}</div>);

  const dn = res ? res.corrupted.detections.length - res.clean.detections.length : 0;
  const m0 = res ? mean(res.clean) : null, m1 = res ? mean(res.corrupted) : null;

  return (<div className="page">
    <div className="pagehead"><div><div className="kicker">Perception Lab</div><h2>See how <span className="hl">conditions change</span> what the model sees</h2></div></div>
    <p className="dim" style={{ maxWidth: 780 }}>The same image runs through the active model twice: clean, and with a <b>simulated</b> corruption. This is a single-image demonstration; dataset-level numbers live in Results.</p>
    <section className="panel labctl">
      <div className={`drop ${drag ? "on" : ""}`} onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) pick(f, f.name); }}>
        <label className="btn primary">Choose image<input hidden type="file" accept="image/*" onChange={e => { const f = e.target.files?.[0]; if (f) pick(f, f.name); e.target.value = ""; }} /></label>
        <span className="dim">{name || "or drop an image here"}</span>
        {samples.length > 0 && <span className="dim small">Samples:</span>}{samples.map(s => <button key={s} className="pillbtn" onClick={() => loadSample(s)}>{s.replace(/\.\w+$/, "")}</button>)}
      </div>
      <div className="chips2">{CORR.map(([k, l]) => <button key={k} className={`pillbtn ${corr === k ? "on" : ""}`} onClick={() => setCorr(k)}>{l}</button>)}</div>
      <div className="bar"><label className="sl grow"><span>Severity</span><b className="mono">{sev}</b><input type="range" min={0} max={5} step={1} value={sev} onChange={e => setSev(+e.target.value)} /></label>
        <label className="chk"><input type="checkbox" checked={auto} onChange={e => setAuto(e.target.checked)} /> Auto-run</label>
        <button className="btn primary" disabled={!blob || busy} onClick={run}>{busy ? "Running…" : "Compare"}</button></div>
      {err && <div className="err static">{err}</div>}
    </section>
    <div className="labgrid">{col("Original", res?.clean ?? null, "clean input")}{col(`Simulated ${CORR.find(c => c[0] === corr)?.[1].toLowerCase()}`, res?.corrupted ?? null, `severity ${sev} of 5 · simulated`)}</div>
    {res && <section className="panel delta"><h3>What changed</h3><div className="deltas">
      <div className="stat"><span>Detections</span><b className="mono">{res.clean.detections.length} → {res.corrupted.detections.length} ({dn >= 0 ? "+" : ""}{dn})</b></div>
      <div className="stat"><span>Mean confidence</span><b className="mono">{m0 == null ? "—" : (m0 * 100).toFixed(1)}% → {m1 == null ? "—" : (m1 * 100).toFixed(1)}%</b></div>
      <div className="stat"><span>Latency</span><b className="mono">{res.clean.infer_ms} → {res.corrupted.infer_ms} ms</b></div></div></section>}
  </div>);
}
