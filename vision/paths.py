"""Where things live.
ROOT = the project folder (code, can be deleted and re-unzipped at any time).
HOME = your persistent data folder (models, datasets, results, uploads, local config).
HOME comes from the SANKETRA_HOME environment variable (scripts\\_env.ps1 sets it for the current terminal session,
default C:\\Users\\<you>\\SanketraData); if it is not set, HOME falls back to ROOT (this is what the tests use)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("SANKETRA_HOME") or ROOT).resolve()


def ensure_home() -> Path:
    for d in ("models", "datasets", "experiments/results", "experiments/runs", "data/uploads"):
        (HOME / d).mkdir(parents=True, exist_ok=True)
    return HOME
