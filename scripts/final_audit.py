"""
User Story:
Provide a self-grading audit that checks whether the repository is submission-ready
against rubric-critical requirements before final handoff.

Implementation Approach:
Run reproducibility, integrity, and documentation checks, then emit a human-readable
SUBMISSION_CHECKLIST.txt with explicit PASS/FAIL outcomes and remediation notes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _check_master_runner(config: str) -> tuple[bool, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, "main.py", "--task", "all", "--config", config]
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return False, "main.py --task all timed out (>300s)."
    ok = p.returncode == 0
    tail = (p.stdout + "\n" + p.stderr)[-1200:]
    if ok:
        return True, "Master runner completed successfully."
    return False, f"Master runner failed (exit {p.returncode}). Output tail:\n{tail}"


def _file_integrity() -> tuple[bool, list[str]]:
    required = [
        ROOT / "docs" / "architecture_final.png",
        ROOT / "results" / "ablation_report.csv",
        ROOT / "dashboard" / "app.py",
        ROOT / "Dockerfile",
        ROOT / "requirements.txt",
        ROOT / "main.py",
        ROOT / "FINAL_REPORT.md",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    return len(missing) == 0, missing


def _doc_header_check() -> tuple[bool, list[str]]:
    bad = []
    for p in ROOT.rglob("*.py"):
        if ".venv" in p.parts:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "User Story:" not in txt or "Implementation Approach:" not in txt:
            bad.append(str(p.relative_to(ROOT)))
    return len(bad) == 0, bad


def write_checklist(path: Path, rows: list[tuple[str, bool, str]]) -> None:
    lines = ["SUBMISSION CHECKLIST", "====================", ""]
    for name, ok, details in rows:
        lines.append(f"[{ 'PASS' if ok else 'FAIL' }] {name}")
        lines.append(details)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run final submission audit and emit checklist.")
    ap.add_argument("--config", type=str, default="config/hybrid_config.yaml")
    ap.add_argument("--out", type=str, default="SUBMISSION_CHECKLIST.txt")
    args = ap.parse_args()

    rows: list[tuple[str, bool, str]] = []

    ok_repro, detail_repro = _check_master_runner(args.config)
    rows.append(("Reproducibility (main.py --task all)", ok_repro, detail_repro))

    ok_files, missing = _file_integrity()
    rows.append(
        (
            "File Integrity",
            ok_files,
            "All required assets found." if ok_files else f"Missing assets: {missing}",
        )
    )

    ok_docs, missing_headers = _doc_header_check()
    rows.append(
        (
            "Documentation Headers (.py scan)",
            ok_docs,
            "All Python files contain required headers."
            if ok_docs
            else f"Missing headers in: {missing_headers[:40]}{' ...' if len(missing_headers) > 40 else ''}",
        )
    )

    overall = all(r[1] for r in rows)
    rows.append(("Overall Submission Readiness", overall, "READY" if overall else "NOT READY"))

    out_path = Path(args.out)
    write_checklist(out_path, rows)
    print(f"Wrote checklist: {out_path}")


if __name__ == "__main__":
    main()
