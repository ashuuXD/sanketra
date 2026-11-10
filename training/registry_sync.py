"""Add registry stubs for dataset classes that have no entry yet (kept in your persistent data folder, never in the project).
Stubs are marked unverified and say the meaning is not entered yet: fill SanketraData\\sign_registry.local.json from the official source.
  python -m training.registry_sync --names <SanketraData>\\datasets\\raw\\indian\\names.txt --source "IRTSD dataset class"
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from vision.paths import HOME, ROOT


def sync(names: list[str], source: str) -> int:
    base = {s["class_name"] for s in json.loads((ROOT / "configs" / "sign_registry.json").read_text(encoding="utf-8"))["signs"]}
    path = HOME / "sign_registry.local.json"
    local = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"signs": []}
    have = base | {s["class_name"] for s in local["signs"]}
    added = 0
    for n in names:
        if n in have:
            continue
        local["signs"].append({"id": f"IN-DS-{n.upper()}", "class_name": n, "label_en": n.replace("_", " ").title(), "label_hi": None,
                               "category": "unknown", "meaning": "Meaning not entered yet. Fill it in from the official source (MVA 1988 First Schedule / IRC:67).",
                               "guidance": "Not entered yet.", "source": source, "verified": False, "needs_review": True, "reference_image": None})
        have.add(n)
        added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(local, indent=2, ensure_ascii=False), encoding="utf-8")
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--names", type=Path, required=True); ap.add_argument("--source", default="dataset class (to be verified)")
    a = ap.parse_args()
    print("registry stubs added:", sync([n.strip() for n in a.names.read_text(encoding="utf-8-sig").splitlines() if n.strip()], a.source), "->", HOME / "sign_registry.local.json")
