export type Det = { track_id: number | null; class_name: string; label_en: string; category: string; raw_conf: number; conf: number; uncertain: boolean; box: number[] };
export type Telemetry = { fps: number; src_fps: number | null; infer_ms: number; e2e_ms: number; n_dets: number; unique_tracks: number; median_ms: number | null; p95_ms: number | null };
export type Ev = { id: number; track_id: number | null; class_name: string; label_en: string; category: string; first_seen: string; last_seen: string; hits: number; best_conf: number; last_conf: number; uncertain: boolean; source: string; thumb: string | null };
export type Sign = { id: string | null; class_name: string; label_en: string; label_hi: string | null; category: string; meaning: string; guidance: string; source: string | null; verified: boolean; in_model?: boolean; needs_review?: boolean };
export type StatusItem = { label: string; state: "done" | "partial" | "todo" | "planned"; hint: string };
export type Settings = { conf: number; imgsz: number; temperature: number; abstain_below: number; tracking: boolean; smoothing: boolean; objects: boolean; pace: boolean };
export type Cam = { index: number; ok: boolean; width: number; height: number; thumb: string | null; error: string | null };
export type Pack = { detections: Det[]; infer_ms: number; image_b64: string };

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) { let m: unknown = r.statusText; try { m = (await r.json()).detail ?? m; } catch { /* ignore */ } throw new Error(typeof m === "string" ? m : JSON.stringify(m)); }
  return r.json();
}
export const api = {
  health: () => fetch("/api/health").then(j<any>),
  model: () => fetch("/api/model").then(j<any>),
  telemetry: () => fetch("/api/telemetry").then(j<any>),
  sources: () => fetch("/api/sources").then(j<{ videos: string[]; stream_active: boolean }>),
  registry: () => fetch("/api/registry").then(j<{ version: string; note: string; signs: Sign[]; model_classes: string[] }>),
  events: (cls = "") => fetch(`/api/events?limit=30&class_name=${encodeURIComponent(cls)}`).then(j<Ev[]>),
  settings: (s: Partial<Settings>) => fetch("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s) }).then(j<Settings>),
  getSettings: () => fetch("/api/settings").then(j<Settings>),
  upload: (path: string, f: Blob | File, extra: Record<string, string> = {}, name = "image.jpg") => { const fd = new FormData(); fd.append("file", f, (f as File).name || name); Object.entries(extra).forEach(([k, v]) => fd.append(k, v)); return fetch(path, { method: "POST", body: fd }).then(j<any>); },
  cameras: () => fetch("/api/cameras?max_index=3").then(j<Cam[]>),
  stop: () => fetch("/api/stream/stop", { method: "POST" }),
  clear: () => fetch("/api/events", { method: "DELETE" }),
  status: () => fetch("/api/status").then(j<StatusItem[]>),
  results: () => fetch("/api/results").then(j<Record<string, any>>),
  samples: () => fetch("/api/samples").then(j<string[]>),
};
