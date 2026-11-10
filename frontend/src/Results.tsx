import { useEffect, useState } from "react";
import { api } from "./api";
import { useReveal } from "./hooks";

const COLS = ["#22d3ee", "#14b8a6", "#f5a524", "#a78bfa", "#94a3b8", "#60a5fa"];
type Series = { name: string; pts: [number, number][] };

function Chart({ series, xl, yl, diag = false, xmax = 1 }: { series: Series[]; xl: string; yl: string; diag?: boolean; xmax?: number }) {
  const W = 620, H = 320, L = 52, R = 16, T = 16, B = 44;
  const X = (x: number) => L + (x / xmax) * (W - L - R), Y = (y: number) => T + (1 - y) * (H - T - B);
  return (<svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={`${yl} versus ${xl}`}>
    {[0, 0.25, 0.5, 0.75, 1].map(v => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke="#1f2933" /><text x={L - 8} y={Y(v) + 4} textAnchor="end" fill="#8b98a5" fontSize="11">{v}</text></g>)}
    {diag && <line x1={X(0)} y1={Y(0)} x2={X(xmax)} y2={Y(1)} stroke="#475569" strokeDasharray="5 5" />}
    {series.map((s, i) => (<g key={s.name}><polyline fill="none" stroke={COLS[i % COLS.length]} strokeWidth="2.2" points={s.pts.map(p => `${X(p[0])},${Y(p[1])}`).join(" ")} />
      {s.pts.map((p, k) => <circle key={k} cx={X(p[0])} cy={Y(p[1])} r="3.4" fill={COLS[i % COLS.length]} />)}</g>))}
    {(xmax === 1 ? [0, 0.25, 0.5, 0.75, 1] : Array.from({ length: xmax + 1 }, (_, i) => i)).map(v => <text key={"x" + v} x={X(v)} y={H - B + 16} textAnchor="middle" fill="#8b98a5" fontSize="11">{v}</text>)}
    <text x={(W + L) / 2} y={H - 8} textAnchor="middle" fill="#8b98a5" fontSize="12">{xl}</text>
    <text transform={`translate(14 ${H / 2}) rotate(-90)`} textAnchor="middle" fill="#8b98a5" fontSize="12">{yl}</text>
    {series.map((s, i) => <g key={"l" + s.name} transform={`translate(${L + 8 + i * 96} ${T + 6})`}><rect width="10" height="10" fill={COLS[i % COLS.length]} /><text x="15" y="9" fill="#e6edf3" fontSize="11">{s.name}</text></g>)}
  </svg>);
}

const CMDS: Record<string, [string, string, string]> = {
  temporal: ["Temporal", "Flicker of detections with and without temporal smoothing (needs a road video)", "python -m evaluation.temporal_eval --model models/best.pt --video your_road_video.mp4 --tag gtsdb_n"],
  detection: ["Detection", "Precision, recall, mAP overall and per class, plus confusion matrix (held-out test split)", "python -m evaluation.detection_report --model models/best.pt --data datasets/prepared/gtsdb/data.yaml --tag gtsdb_n"],
  robustness: ["Robustness", "mAP versus corruption severity", "python -m evaluation.robustness --model models/best.pt --data datasets/prepared/gtsdb/data.yaml --split test"],
  calibration: ["Calibration", "Reliability and ECE, before and after temperature scaling", "python -m evaluation.calibration_eval --model models/best.pt --data datasets/prepared/gtsdb/data.yaml"],
  latency: ["Latency", "Median and p95 latency, FPS and model size on this machine", "python -m evaluation.benchmark_latency --models models/best.pt --imgsz 320 416 640"],
};
const Cmd = ({ c }: { c: string }) => { const [ok, setOk] = useState(false); return (<div className="cmd mono small"><code>{c}</code><button className="btn" onClick={() => { navigator.clipboard?.writeText(c); setOk(true); setTimeout(() => setOk(false), 1200); }}>{ok ? "Copied" : "Copy"}</button></div>); };

export default function Results() {
  const [r, setR] = useState<Record<string, any> | null>(null); const [err, setErr] = useState("");
  useEffect(() => { api.results().then(setR).catch(e => setErr(String(e))); }, []);
  useReveal(r);
  if (err) return <div className="err static">{err}</div>;
  if (!r) return <p className="dim pad">Loading…</p>;
  const keys = Object.keys(r);
  const find = (p: string) => keys.filter(k => k.startsWith(p));
  return (<div className="page">
    <div className="pagehead"><div><div className="kicker">Results</div><h2>Only <span className="hl">measured</span> numbers appear here</h2></div></div>
    <p className="dim" style={{ maxWidth: 760 }}>Each card reads a file produced by an evaluation script. Nothing is simulated or pre-filled. A card stays empty until you run its experiment.</p>
    <div className="cards3">{Object.entries(CMDS).map(([k, [t, d, cmd]], i) => { const n = find(k).length; return (
      <div key={k} className={`reveal card d${i}`}><div className="row"><h3>{t}</h3><span className={`chip ${n ? "ok" : "warn"}`}>{n ? `${n} result file${n > 1 ? "s" : ""}` : "not run yet"}</span></div><p className="dim small">{d}</p>{!n && <Cmd c={cmd} />}</div>); })}</div>

    {find("detection").length >= 2 && (<section className="panel reveal"><h3>Model comparison</h3>
      <p className="dim small">Same held-out split. Check that the models were trained with the same recipe (epochs, data, image size) before reading a difference as caused by the architecture.</p>
      <table><thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th><th>Median ms @640</th><th>Size MB</th></tr></thead><tbody>
        {find("detection").map(k => { const o = r[k]; if (!o?.overall) return null; const lat = r[k.replace("detection", "latency")]?.rows?.find((x: any) => x.imgsz === 640);
          return <tr key={k}><td>{o.model}</td><td className="mono">{o.overall.precision.toFixed(3)}</td><td className="mono">{o.overall.recall.toFixed(3)}</td><td className="mono">{o.overall.map50.toFixed(3)}</td><td className="mono">{o.overall.map50_95.toFixed(3)}</td><td className="mono">{lat ? lat.median_ms : "—"}</td><td className="mono">{lat ? lat.size_mb : "—"}</td></tr>; })}
      </tbody></table></section>)}

    {find("detection").map(k => { const o = r[k]; if (!o?.overall) return null; const f = (v: number) => v.toFixed(3);
      return (<details key={k} className="panel reveal"><summary><h3 style={{ display: "inline" }}>{k}</h3></summary><p className="dim small">{o.model} on the {o.split} split at image size {o.imgsz}. One training run, one seed: treat differences of a few points as noise.</p>
        <div className="deltas"><div className="stat"><span>Precision</span><b className="mono">{f(o.overall.precision)}</b></div><div className="stat"><span>Recall</span><b className="mono">{f(o.overall.recall)}</b></div><div className="stat"><span>mAP@0.5</span><b className="mono">{f(o.overall.map50)}</b></div><div className="stat"><span>mAP@0.5:0.95</span><b className="mono">{f(o.overall.map50_95)}</b></div></div>
        <table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th></tr></thead><tbody>{o.per_class.map((c: any) => <tr key={c.class}><td>{c.class}</td><td className="mono">{f(c.precision)}</td><td className="mono">{f(c.recall)}</td><td className="mono">{f(c.map50)}</td><td className="mono">{f(c.map50_95)}</td></tr>)}</tbody></table>
        {o.confusion_png && <><p className="dim small" style={{ marginTop: 12 }}>Normalised confusion matrix (rows: predicted, columns: true; last row/column is background).</p><img className="plot" alt="confusion matrix" src={`/api/results/${o.confusion_png}.png`} /></>}</details>); })}

    {find("temporal").length > 0 && (<section className="panel reveal"><h3>Temporal smoothing</h3>
      <p className="dim small">Flicker rate = share of consecutive frames where a tracked box appears or disappears; lower is steadier. Only tracks seen in at least 10 frames count. This measures stability, not correctness. Runs with fewer than 10 tracks are dimmed: too few to read anything from.</p>
      <table><thead><tr><th>Run</th><th>Model</th><th>Video</th><th>Tracks</th><th>Raw</th><th>EMA</th><th>EMA + hold 3</th></tr></thead><tbody>
        {find("temporal").map(k => { const o = r[k]; if (!o?.flicker_rate) return null; const f = o.flicker_rate; const few = (o.tracks_analysed ?? 0) < 10; const fm = (v: number | null) => (v == null ? "—" : v.toFixed(3));
          return <tr key={k} style={few ? { opacity: 0.5 } : undefined}><td className="mono small">{k.replace("temporal_", "")}</td><td>{o.model}</td><td>{o.video}</td><td className="mono">{o.tracks_analysed}</td><td className="mono">{fm(f.raw)}</td><td className="mono">{fm(f.ema)}</td><td className="mono">{fm(f.ema_hold3)}</td></tr>; })}
      </tbody></table></section>)}

    {find("robustness").map(k => { const o = r[k]; if (!o?.results) return null;
      const series: Series[] = Object.entries<any>(o.results).map(([c, per]) => ({ name: c, pts: Object.keys(per).map(Number).sort((a, b) => a - b).map(s => [s, per[String(s)].map50] as [number, number]) }));
      return (<details key={k} className="panel reveal"><summary><h3 style={{ display: "inline" }}>{k}</h3></summary><p className="dim small">mAP@0.5 on the {o.split} split at image size {o.imgsz}. Corruptions are simulated (severity 0 = clean).</p><Chart series={series} xl="corruption severity" yl="mAP@0.5" xmax={5} /></details>); })}

    {find("calibration").map(k => { const o = r[k]; if (!o?.bins_before) return null;
      const mk = (b: any[], n: string): Series => ({ name: n, pts: b.filter(x => x.n).map(x => [x.conf, x.acc] as [number, number]) });
      return (<details key={k} className="panel reveal"><summary><h3 style={{ display: "inline" }}>{k}</h3></summary><p className="dim small">Points on the dashed diagonal mean displayed confidence matches observed precision. Test split; temperature fitted on validation only.</p>
        <Chart series={[mk(o.bins_before, "raw"), mk(o.bins_after, "calibrated")]} xl="confidence" yl="precision in bin" diag />
        <div className="deltas"><div className="stat"><span>ECE before</span><b className="mono">{o.ece_before.toFixed(3)}</b></div><div className="stat"><span>ECE after</span><b className="mono">{o.ece_after.toFixed(3)}</b></div><div className="stat"><span>Temperature</span><b className="mono">{o.temperature_fitted_on_val.toFixed(2)}</b></div></div></details>); })}

    {find("latency").map(k => { const o = r[k]; if (!o?.rows) return null; const mx = Math.max(...o.rows.map((x: any) => x.median_ms));
      return (<details key={k} className="panel reveal"><summary><h3 style={{ display: "inline" }}>{k}</h3></summary><p className="dim small">{o.machine} · device {o.device}</p>
        <table><thead><tr><th>Model</th><th>imgsz</th><th>Median ms</th><th>p95 ms</th><th>FPS</th><th>Size MB</th><th></th></tr></thead><tbody>
          {o.rows.map((x: any, i: number) => <tr key={i}><td>{x.model}</td><td>{x.imgsz}</td><td className="mono">{x.median_ms}</td><td className="mono">{x.p95_ms}</td><td className="mono">{x.fps_from_median}</td><td className="mono">{x.size_mb}</td><td style={{ width: "28%" }}><div className="meter"><div style={{ width: `${(x.median_ms / mx) * 100}%` }} /></div></td></tr>)}</tbody></table></details>); })}
  </div>);
}
