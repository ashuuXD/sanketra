import { useEffect, useRef, useState } from "react";

/** Adds class "in" to every .reveal element once it scrolls into view. */
export function useReveal(dep: unknown = null) {
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>(".reveal:not(.in)");
    if (!("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("in")); return; }
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.12 });
    els.forEach(e => io.observe(e));
    return () => io.disconnect();
  }, [dep]);
}

/** Writes scroll progress (0..1) of an element into its CSS variable --p. No React state, so scrolling never re-renders. */
export function useScrollVar<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  useEffect(() => {
    let raf = 0;
    const on = () => {
      raf = 0; const el = ref.current; if (!el) return;
      const r = el.getBoundingClientRect(), vh = window.innerHeight;
      el.style.setProperty("--p", String(Math.max(0, Math.min(1, (vh * 0.65 - r.top) / (r.height || 1)))));
    };
    const sched = () => { if (!raf) raf = requestAnimationFrame(on); };
    on(); window.addEventListener("scroll", sched, { passive: true }); window.addEventListener("resize", sched);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("scroll", sched); window.removeEventListener("resize", sched); };
  }, []);
  return ref;
}

/** True once the element has entered the viewport (fires once). */
export function useInView<T extends HTMLElement>() {
  const ref = useRef<T | null>(null); const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current; if (!el || !("IntersectionObserver" in window)) { setSeen(true); return; }
    const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { setSeen(true); io.disconnect(); } }, { threshold: 0.3 });
    io.observe(el); return () => io.disconnect();
  }, []);
  return [ref, seen] as const;
}

export function useCount(target: number, run: boolean, ms = 900) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!run) return; let raf = 0; const t0 = performance.now();
    const tick = (t: number) => { const k = Math.min(1, (t - t0) / ms); setV(Math.round(target * (1 - Math.pow(1 - k, 3)))); if (k < 1) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [target, run, ms]);
  return v;
}
