import { useEffect, useRef } from "react";

/** Decorative ambient background: perspective road grid, drifting sign reticles, dust. Not live detections. */
export default function Fx({ intensity, animate }: { intensity: "hero" | "ambient"; animate: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!; const ctx = c.getContext("2d")!;
    const hero = intensity === "hero";
    const still = !animate || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, raf = 0, last = 0, t = 0, mx = 0.5, my = 0.5, started = false;
    const dpr = 1;
    let A = hero ? 1 : 0.45;
    const resize = () => { w = window.innerWidth; h = window.innerHeight; c.width = w * dpr; c.height = h * dpr; c.style.width = w + "px"; c.style.height = h + "px"; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); if (still && started) { last = 0; raf = requestAnimationFrame(frame); } };
    resize();
    const rnd = (a: number, b: number) => a + Math.random() * (b - a);
    const reticles = Array.from({ length: hero ? 7 : 3 }, () => ({ lat: rnd(-1, 1), z: Math.random(), v: rnd(0.05, 0.09), k: Math.floor(rnd(0, 3)) }));
    const dust = Array.from({ length: hero ? 60 : 24 }, () => ({ x: Math.random(), y: Math.random(), s: rnd(0.4, 1.4), v: rnd(0.004, 0.015) }));
    const onMove = (e: MouseEvent) => { mx = e.clientX / w; my = e.clientY / h; };
    window.addEventListener("mousemove", onMove, { passive: true }); window.addEventListener("resize", resize);

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const heroNow = hero;                                              // full effect at every scroll position on Home
      if (document.hidden || now - last < (hero ? 40 : 90)) return;
      const dt = Math.min(0.1, (now - last) / 1000); last = now; if (!still) t += dt;
      const vx = w * (0.5 + (mx - 0.5) * 0.04), hy = h * (hero ? 0.44 : 0.38) + (my - 0.5) * 8;
      ctx.clearRect(0, 0, w, h);
      const g = ctx.createRadialGradient(vx, hy, 0, vx, hy, Math.max(w, h) * 0.6);
      g.addColorStop(0, `rgba(34,211,238,${0.16 * A})`); g.addColorStop(0.5, `rgba(20,184,166,${0.05 * A})`); g.addColorStop(1, "rgba(10,13,16,0)");
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
      // road lanes converging to the vanishing point
      ctx.lineWidth = 1;
      for (let k = -7; k <= 7; k++) {
        const bx = vx + k * w * 0.2; const edge = Math.abs(k) === 1;
        ctx.strokeStyle = `rgba(34,211,238,${(edge ? 0.34 : 0.12) * A})`;
        ctx.beginPath(); ctx.moveTo(vx, hy); ctx.lineTo(bx, h); ctx.stroke();
      }
      // cross lines rushing toward the viewer
      for (let i = 0; i < 16; i++) {
        const z = ((i / 16 + t * 0.12) % 1); const y = hy + (h - hy) * z * z;
        ctx.strokeStyle = `rgba(34,211,238,${0.05 * A + 0.14 * A * z})`;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      if (!heroNow) { if (still) cancelAnimationFrame(raf); return; }   // working screens: soft glow + lanes only
      // centre dashes
      ctx.fillStyle = `rgba(230,237,243,${0.28 * A})`;
      for (let i = 0; i < 14; i++) {
        const z0 = ((i / 14 + t * 0.2) % 1), z1 = z0 + 0.025; const y0 = hy + (h - hy) * z0 * z0, y1 = hy + (h - hy) * z1 * z1;
        const wd0 = 1 + z0 * 7, wd1 = 1 + z1 * 7; ctx.beginPath(); ctx.moveTo(vx - wd0, y0); ctx.lineTo(vx + wd0, y0); ctx.lineTo(vx + wd1, y1); ctx.lineTo(vx - wd1, y1); ctx.fill();
      }
      // sign reticles: corner brackets that grow as they approach
      for (const r of reticles) {
        if (!still) r.z += r.v * dt; if (r.z > 1) { r.z = 0.02; r.lat = rnd(-1, 1); r.k = Math.floor(rnd(0, 3)); }
        const zz = r.z * r.z, s = 8 + zz * 110, x = vx + r.lat * w * 0.55 * r.z, y = hy + (h - hy) * zz * 0.8 - s * 0.6;
        const a = Math.sin(Math.PI * Math.min(1, r.z * 1.05)) * 0.8 * A; ctx.strokeStyle = `rgba(34,211,238,${a})`; ctx.lineWidth = 1.5;
        const L = s * 0.28; ctx.beginPath();
        for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) { ctx.moveTo(x + sx * s, y + sy * (s - L)); ctx.lineTo(x + sx * s, y + sy * s); ctx.lineTo(x + sx * (s - L), y + sy * s); }
        ctx.stroke();
        ctx.strokeStyle = `rgba(230,237,243,${a * 0.55})`; ctx.lineWidth = 1.2; ctx.beginPath();
        if (r.k === 0) ctx.arc(x, y, s * 0.55, 0, 6.283); else if (r.k === 1) { ctx.moveTo(x, y - s * 0.6); ctx.lineTo(x + s * 0.6, y + s * 0.5); ctx.lineTo(x - s * 0.6, y + s * 0.5); ctx.closePath(); } else ctx.rect(x - s * 0.5, y - s * 0.5, s, s);
        ctx.stroke();
      }
      // dust
      ctx.fillStyle = `rgba(160,235,245,${0.35 * A})`;
      for (const d of dust) { if (!still) d.y -= d.v * dt * 4; if (d.y < 0) { d.y = 1; d.x = Math.random(); } ctx.fillRect(d.x * w, d.y * h, d.s, d.s); }
      if (still) cancelAnimationFrame(raf);
    };
    started = true; raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("mousemove", onMove); window.removeEventListener("resize", resize); };
  }, [intensity]);
  return <canvas ref={ref} className="fx" aria-hidden />;
}
