from __future__ import annotations
import os
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from vision.paths import HOME, ROOT  # noqa: E402
DISCLAIMER = "Educational prototype. Not a certified automotive safety system. Do not rely on it for driving decisions."


def load_config(path: Path | None = None) -> dict:
    path = path or Path(os.environ.get("SANKETRA_CONFIG", ROOT / "configs" / "default.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = HOME / "config.local.yaml"          # your persistent overrides (written by the training script)
    if local.exists():
        for section, vals in (yaml.safe_load(local.read_text(encoding="utf-8-sig")) or {}).items():
            cfg.setdefault(section, {}).update(vals or {})
    if os.environ.get("SANKETRA_MODEL"):
        cfg["model"]["path"] = os.environ["SANKETRA_MODEL"]
    if os.environ.get("SANKETRA_DEVICE"):
        cfg["model"]["device"] = os.environ["SANKETRA_DEVICE"]
    return cfg


class RuntimeSettings(BaseModel):
    """User-adjustable inference settings (validated)."""
    conf: float = Field(0.25, ge=0.05, le=0.95)
    imgsz: int = Field(640)
    temperature: float = Field(1.0, gt=0.1, le=5.0)
    abstain_below: float = Field(0.5, ge=0.0, le=1.0)
    tracking: bool = True
    smoothing: bool = True
    objects: bool = True         # also detect people and vehicles with a second, general model (if available)
    pace: bool = True            # pace video playback to the source FPS (off = measure raw throughput)

    def model_post_init(self, _):
        if self.imgsz not in (320, 416, 640):
            raise ValueError("imgsz must be one of 320, 416, 640")

    @classmethod
    def from_config(cls, cfg: dict) -> "RuntimeSettings":
        i = cfg["inference"]
        return cls(conf=i["conf"], imgsz=i["imgsz"], temperature=i["temperature"],
                   abstain_below=i["abstain_below"], tracking=i["tracking"], smoothing=i["smoothing"])
