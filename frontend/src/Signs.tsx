import { useEffect, useState } from "react";
import { api, Sign } from "./api";

export default function Signs() {
  const [d, setD] = useState<{ version: string; note: string; signs: Sign[]; model_classes: string[] } | null>(null);
  const [q, setQ] = useState(""); const [cat, setCat] = useState("all"); const [err, setErr] = useState("");
  useEffect(() => { api.registry().then(setD).catch(e => setErr(String(e))); }, []);
  if (err) return <div className="err static">{err}</div>;
  if (!d) return <p className="dim pad">Loading…</p>;
  const cats = ["all", ...Array.from(new Set(d.signs.map(s => s.category)))];
  const list = d.signs.filter(s => (cat === "all" || s.category === cat) && (s.label_en + s.meaning).toLowerCase().includes(q.toLowerCase()));
  return (<div className="page">
    <div className="pagehead"><div><div className="kicker">Sign registry · v{d.version}</div><h2>What each sign <span className="hl">means</span></h2></div>
      <div className="bar"><input className="input" placeholder="Search signs…" value={q} onChange={e => setQ(e.target.value)} />
        {cats.map(c => <button key={c} className={`pillbtn ${cat === c ? "on" : ""}`} onClick={() => setCat(c)}>{c}</button>)}</div></div>
    <p className="dim small">{d.signs.filter(x => x.in_model).length} of {d.signs.length} registry entries are detected by the active model ({d.model_classes.length} model classes). Entries marked “meaning to fill in” were imported from a dataset: add the official meaning in <span className="mono">SanketraData\\sign_registry.local.json</span>.</p>
    <p className="warnbox">{d.note}</p>
    <div className="signgrid">{list.map((s, i) => (<div key={s.class_name} className="card fadein" style={{ animationDelay: `${Math.min(i, 8) * 40}ms` }}>
      <div className="row"><span className={`cat cat-${s.category}`}>{s.category}</span><span className={`chip ${s.in_model ? "ok" : "dim"}`} title="Can the active detector find this sign?">{s.in_model ? "detected by current model" : "not in current model"}</span>{s.needs_review && <span className="chip warn">meaning to fill in</span>}<span className={`chip ${s.verified ? "ok" : "warn"}`}>{s.verified ? "verified" : "unverified draft"}</span></div>
      <h3>{s.label_en}</h3><p>{s.meaning}</p><p className="dim small">{s.guidance}</p>
      <div className="dim small mono">model class: {s.class_name}{s.label_hi ? ` · ${s.label_hi}` : ""}</div><div className="dim small">{s.source}</div></div>))}
      {list.length === 0 && <p className="dim">No signs match.</p>}</div>
  </div>);
}
