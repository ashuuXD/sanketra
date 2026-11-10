import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Wordmark } from "./Brand";
import Fx from "./Fx";
import Home from "./Home";
import Console from "./Console";
import Lab from "./Lab";
import Results from "./Results";
import Signs from "./Signs";

type Tab = "home" | "console" | "lab" | "results" | "signs";
const TABS: [Tab, string][] = [["home", "Home"], ["console", "Console"], ["lab", "Perception Lab"], ["results", "Results"], ["signs", "Signs"]];

function Splash({ onDone }: { onDone: () => void }) {
  const [steps, setSteps] = useState<{ t: string; s: "wait" | "ok" | "err" }[]>([{ t: "API reachable", s: "wait" }, { t: "Model initialised", s: "wait" }, { t: "Input sources checked", s: "wait" }]);
  const [msg, setMsg] = useState(""); const [sec, setSec] = useState(0);
  useEffect(() => { const t = setInterval(() => setSec(x => x + 1), 1000); return () => clearInterval(t); }, []);
  const set = (i: number, s: "ok" | "err") => setSteps(p => p.map((x, k) => (k === i ? { ...x, s } : x)));
  useEffect(() => {
    let dead = false;
    (async () => {
      for (let n = 0; n < 180 && !dead; n++) {
        try {
          const h = await api.health(); set(0, "ok");
          if (h.model_status === "ready") {
            const m = await api.model(); set(1, "ok"); let line = `${m.name} · ${m.device} · ${m.params_million}M params`; setMsg(line);
            const s = await api.sources(); set(2, "ok"); setMsg(`${line} · ${s.videos.length} video(s) · camera chosen in Console`);
            setTimeout(onDone, 600); return;
          }
          if (h.model_status === "error") { set(1, "err"); setMsg(h.model_error); return; }
        } catch { /* API not up yet */ }
        await new Promise(r => setTimeout(r, 1000));
      }
      if (!dead) setMsg("Backend not reachable. Start it with scripts\\run.ps1");
    })();
    return () => { dead = true; };
  }, [onDone]);
  return (<div className="splash"><div className="scan" /><Wordmark size={42} />
    <ul>{steps.map(s => <li key={s.t} className={s.s}><span className="dot" />{s.t}</li>)}</ul><p className="mono dim small">{msg || `Starting backend and loading the model… ${sec}s`}</p></div>);
}

export default function App() {
  const [ready, setReady] = useState(false);
  const [stale, setStale] = useState(false);
  useEffect(() => {
    const mine = document.querySelector('script[type="module"][src*="/assets/"]')?.getAttribute("src");
    if (!mine) return;                                   // dev server: nothing to compare
    const check = () => api.health().then(h => { if (h.ui_build && h.ui_build !== mine) setStale(true); }).catch(() => {});
    check(); const t = setInterval(check, 15000); return () => clearInterval(t);
  }, []);
  const [tab, setTab] = useState<Tab>(() => (TABS.map(t => t[0]).includes(location.hash.slice(1) as Tab) ? (location.hash.slice(1) as Tab) : "home"));
  const [fx, setFx] = useState(() => localStorage.getItem("sanketra_fx") !== "off");
  const go = useCallback((t: Tab) => { setTab(t); location.hash = t; window.scrollTo({ top: 0 }); }, []);
  useEffect(() => { const on = () => { const t = location.hash.slice(1) as Tab; if (TABS.some(x => x[0] === t)) setTab(t); }; window.addEventListener("hashchange", on); return () => window.removeEventListener("hashchange", on); }, []);
  const toggleFx = () => setFx(v => { localStorage.setItem("sanketra_fx", v ? "off" : "on"); return !v; });
  if (!ready) return <Splash onDone={() => setReady(true)} />;
  return (<div className="app">
    {fx && <Fx key={tab === "home" ? "h" : "s"} intensity={tab === "home" ? "hero" : "ambient"} animate={tab === "home"} />}
    <header><button className="brandbtn" onClick={() => go("home")} aria-label="Home"><Wordmark /></button>
      <nav>{TABS.map(([k, l]) => <button key={k} className={tab === k ? "on" : ""} onClick={() => go(k)}>{l}</button>)}</nav>
      <button className={`fxbtn ${fx ? "on" : ""}`} onClick={toggleFx} title="Ambient background animation (decorative)">FX {fx ? "on" : "off"}</button></header>
    {stale && <div className="stale">A newer version of this interface is available. <button className="btn" onClick={() => location.reload()}>Reload now</button></div>}
    <main className={tab === "home" ? "wide" : ""}>
      {tab === "home" ? <Home go={go} /> : tab === "console" ? <Console /> : tab === "lab" ? <Lab /> : tab === "results" ? <Results /> : <Signs />}
    </main>
    <footer>Educational prototype. Not a certified automotive safety system. Detections can be wrong, even at high confidence.</footer>
  </div>);
}
