from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # backend/src/paths.py -> backend/src -> backend -> repo_root
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
DB_PATH = ROOT / "app.db"
LOGS_DIR = ROOT / "logs"
RUNS_DIR = ROOT / "results" / "runs"

