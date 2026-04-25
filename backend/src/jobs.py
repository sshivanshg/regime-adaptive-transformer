from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.src import db
from backend.src.metrics import compute_headline_from_files
from backend.src.paths import LOGS_DIR, ROOT, RUNS_DIR


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobSpec:
    run_id: str
    run_type: str
    params: dict[str, Any]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def submit(self, run_type: str, params: dict[str, Any]) -> str:
        run_id = str(uuid4())
        spec = JobSpec(run_id=run_id, run_type=run_type, params=params)
        db.create_run(run_id=run_id, run_type=run_type, params=params)

        t = threading.Thread(target=self._run_job, args=(spec,), daemon=True)
        with self._lock:
            self._threads[run_id] = t
        t.start()
        return run_id

    def _run_job(self, spec: JobSpec) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        stdout_path = LOGS_DIR / f"run_{spec.run_id}_stdout.log"
        stderr_path = LOGS_DIR / f"run_{spec.run_id}_stderr.log"
        artifacts_dir = RUNS_DIR / spec.run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        db.update_run_status(
            spec.run_id,
            status="running",
            started_at=_utc_now_iso(),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

        try:
            if spec.run_type == "fetch-latest":
                self._job_fetch_latest(spec, stdout_path, stderr_path)
                self._collect_fetch_artifacts(spec, artifacts_dir)
            elif spec.run_type == "run-monthly":
                self._job_run_monthly(spec, stdout_path, stderr_path)
                self._collect_monthly_artifacts(spec, artifacts_dir)
            elif spec.run_type == "backtest":
                self._job_backtest_year(spec, stdout_path, stderr_path, artifacts_dir)
                self._collect_backtest_artifacts(spec, artifacts_dir)
            else:
                raise ValueError(f"Unknown run_type: {spec.run_type}")

            db.update_run_status(spec.run_id, status="succeeded", finished_at=_utc_now_iso())
        except Exception as e:
            db.update_run_status(
                spec.run_id,
                status="failed",
                finished_at=_utc_now_iso(),
                error_message=f"{type(e).__name__}: {e}",
            )
            raise

    def _run_subprocess(
        self,
        cmd: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        with open(stdout_path, "ab") as out_f, open(stderr_path, "ab") as err_f:
            out_f.write((f"\n[{_utc_now_iso()}] $ {' '.join(cmd)}\n").encode("utf-8"))
            out_f.flush()
            merged_env = dict(os.environ)
            merged_env["PYTHONUNBUFFERED"] = "1"
            if env:
                merged_env.update(env)
            p = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=merged_env,
                stdout=out_f,
                stderr=err_f,
                check=False,
            )
            if p.returncode != 0:
                raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")

    def _job_fetch_latest(self, spec: JobSpec, stdout_path: Path, stderr_path: Path) -> None:
        start = str(spec.params.get("start", "2020-01-01"))
        end_excl = str(spec.params["end_date_exclusive"])
        sleep_s = str(spec.params.get("sleep_s", 0.5))

        cmd = [
            sys.executable,
            "scripts/fetch_nifty200.py",
            "--start",
            start,
            "--end",
            end_excl,
            "--raw-dir",
            "data/raw",
            "--index",
            "nifty200",
            "--sleep",
            sleep_s,
        ]
        self._run_subprocess(cmd, stdout_path=stdout_path, stderr_path=stderr_path, cwd=ROOT)

    def _job_run_monthly(self, spec: JobSpec, stdout_path: Path, stderr_path: Path) -> None:
        end_excl = str(spec.params["end_date_exclusive"])
        start = str(spec.params.get("start", "2020-01-01"))
        self._run_subprocess(
            [
                sys.executable,
                "-m",
                "backend.src.pipeline.run_features",
                "--raw-dir",
                "data/raw",
                "--processed-dir",
                "data/processed",
                "--start",
                start,
                "--end",
                end_excl,
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=ROOT,
        )
        self._run_subprocess(
            [sys.executable, "scripts/build_momentum_predictions.py"],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=ROOT,
        )
        self._run_subprocess(
            [sys.executable, "models/run_final_2024_2026.py", "--backtest-only"],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=ROOT,
        )

    def _job_backtest_year(
        self,
        spec: JobSpec,
        stdout_path: Path,
        stderr_path: Path,
        artifacts_dir: Path,
    ) -> None:
        # For now, we run a year-scoped backtest without overwriting production outputs by
        # copying current outputs into the run folder and then running the backtest-only path
        # against that copied predictions file. In a later pass we can implement a fully
        # isolated runner that writes to run-scoped locations from the start.
        year = int(spec.params["year"])
        preds_src = ROOT / "results" / "final_strategy" / "ranking_predictions.csv"
        if not preds_src.is_file():
            raise FileNotFoundError(f"Missing predictions CSV: {preds_src}")

        preds_dst = artifacts_dir / "ranking_predictions.csv"
        shutil.copy2(preds_src, preds_dst)

        cmd = [
            sys.executable,
            "models/run_final_2024_2026.py",
            "--backtest-only",
            "--predictions",
            str(preds_dst),
        ]
        self._run_subprocess(cmd, stdout_path=stdout_path, stderr_path=stderr_path, cwd=ROOT)

        # Filter the produced backtest to the requested year and save alongside.
        bt_src = ROOT / "results" / "final_strategy" / "backtest_results.csv"
        if bt_src.is_file():
            import pandas as pd  # local import keeps backend lightweight at import time

            bt = pd.read_csv(bt_src, parse_dates=["date"])
            bt_y = bt[bt["date"].dt.year == year].copy()
            bt_y.to_csv(artifacts_dir / f"backtest_results_{year}.csv", index=False)

    def _collect_fetch_artifacts(self, spec: JobSpec, artifacts_dir: Path) -> None:
        stats = ROOT / "data" / "raw" / "_fetch_stats.json"
        if stats.is_file():
            dst = artifacts_dir / stats.name
            shutil.copy2(stats, dst)
            db.add_artifact(spec.run_id, str(dst), kind="fetch_stats")
        self._update_latest_pointer("latest_fetch", artifacts_dir)

    def _collect_monthly_artifacts(self, spec: JobSpec, artifacts_dir: Path) -> None:
        for rel in [
            "results/final_strategy/ranking_predictions.csv",
            "results/final_strategy/monthly_rankings.csv",
            "results/final_strategy/backtest_results.csv",
        ]:
            p = ROOT / rel
            if p.is_file():
                dst = artifacts_dir / Path(rel).name
                shutil.copy2(p, dst)
                db.add_artifact(spec.run_id, str(dst), kind="result")

        bt = artifacts_dir / "backtest_results.csv"
        nifty = ROOT / "data" / "raw" / "_NSEI.parquet"
        if bt.is_file() and nifty.is_file():
            hm = compute_headline_from_files(backtest_csv=bt, nifty_parquet=nifty)
            db.upsert_headline_metrics(spec.run_id, hm.as_dict())

            # Persist latest picks (top names at the last rebalance) using the monthly_rankings export.
            try:
                import pandas as pd

                bt_df = pd.read_csv(bt, parse_dates=["date"])
                last_date = pd.to_datetime(bt_df["date"].max()).strftime("%Y-%m-%d")
                mr = artifacts_dir / "monthly_rankings.csv"
                if mr.is_file():
                    rk = pd.read_csv(mr, parse_dates=["Date"])
                    sub = rk[rk["Date"] == pd.Timestamp(last_date)].copy()
                    if not sub.empty:
                        sub = sub.sort_values("score", ascending=False).head(5)
                        picks = []
                        for _, row in sub.iterrows():
                            picks.append(
                                {
                                    "ticker": str(row["Ticker"]),
                                    "weight": None,
                                    "regime": None,
                                    "momentum": float(row["momentum"]) if "momentum" in row else None,
                                }
                            )
                        db.replace_picks(spec.run_id, last_date, picks)
            except Exception:
                # Metrics are still valid; picks are best-effort.
                pass
        self._update_latest_pointer("latest_monthly", artifacts_dir)

    def _collect_backtest_artifacts(self, spec: JobSpec, artifacts_dir: Path) -> None:
        for p in artifacts_dir.glob("*.csv"):
            db.add_artifact(spec.run_id, str(p), kind="result")

    def _update_latest_pointer(self, name: str, artifacts_dir: Path) -> None:
        """
        Maintain a convenient, non-versioned pointer directory under results/runs/<name>/.
        We use a copy (not symlink) for portability.
        """
        target = RUNS_DIR / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(artifacts_dir, target)


JOB_MANAGER = JobManager()

