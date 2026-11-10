from __future__ import annotations
import json
from pathlib import Path
from .settings import HOME, ROOT

_UNKNOWN = {"id": None, "label_hi": None, "category": "unknown", "meaning": "No registry entry for this class.",
            "guidance": "Treat as an unverified detection.", "source": None, "verified": False, "reference_image": None}


class SignRegistry:
    def __init__(self, path: Path | None = None):
        path = path or ROOT / "configs" / "sign_registry.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version, self.note = data["version"], data.get("note", "")
        self.by_class = {s["class_name"]: s for s in data["signs"]}
        local = HOME / "sign_registry.local.json"            # your own / imported entries; survive re-unzipping the project
        if local.exists():
            try:
                for s in json.loads(local.read_text(encoding="utf-8-sig")).get("signs", []):
                    self.by_class[s["class_name"]] = {**self.by_class.get(s["class_name"], {}), **s}
            except (OSError, ValueError, KeyError) as e:
                self.note += f" (could not read {local.name}: {e})"

    def lookup(self, class_name: str) -> dict:
        if class_name in self.by_class:
            return self.by_class[class_name]
        return {**_UNKNOWN, "class_name": class_name, "label_en": class_name.replace("_", " ").title()}

    def all(self) -> list[dict]:
        return list(self.by_class.values())
