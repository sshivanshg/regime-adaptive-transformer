from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend.src import db
from backend.src.jobs import JOB_MANAGER

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _default_end_date_exclusive() -> str:
    # yfinance "end" is exclusive, so today+1 includes today if markets traded.
    return (date.today() + timedelta(days=1)).isoformat()


class RunFetchLatestRequest(BaseModel):
    start: str = Field(default="2020-01-01")
    end_date_exclusive: str | None = Field(default=None)
    sleep_s: float = Field(default=0.5, ge=0.0, le=10.0)


@router.post("/jobs/fetch-latest")
def jobs_fetch_latest(req: RunFetchLatestRequest) -> dict[str, Any]:
    params = req.model_dump()
    if not params.get("end_date_exclusive"):
        params["end_date_exclusive"] = _default_end_date_exclusive()
    run_id = JOB_MANAGER.submit("fetch-latest", params)
    return {"job_id": run_id}


class RunMonthlyRequest(BaseModel):
    end_date_exclusive: str | None = Field(default=None)


@router.post("/jobs/run-monthly")
def jobs_run_monthly(req: RunMonthlyRequest) -> dict[str, Any]:
    params = req.model_dump()
    if not params.get("end_date_exclusive"):
        params["end_date_exclusive"] = _default_end_date_exclusive()
    run_id = JOB_MANAGER.submit("run-monthly", params)
    return {"job_id": run_id}


class BacktestRequest(BaseModel):
    year: int = Field(ge=1990, le=2100)


@router.post("/jobs/backtest")
def jobs_backtest(req: BacktestRequest) -> dict[str, Any]:
    run_id = JOB_MANAGER.submit("backtest", req.model_dump())
    return {"job_id": run_id}


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    r = db.get_run(job_id)
    if r is None:
        raise HTTPException(status_code=404, detail="job not found")
    r["artifacts"] = db.list_artifacts(job_id)
    return r


@router.get("/jobs/{job_id}/artifacts")
def job_artifacts(job_id: str) -> dict[str, Any]:
    r = db.get_run(job_id)
    if r is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "artifacts": db.list_artifacts(job_id)}


@router.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
def job_logs(job_id: str, stream: str = "stdout", tail: int = 300) -> str:
    r = db.get_run(job_id)
    if r is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = r.get("stdout_path") if stream == "stdout" else r.get("stderr_path")
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        n = max(1, min(int(tail), 5000))
        return "\n".join(lines[-n:]) + "\n"
    except Exception:
        return ""


@router.get("/results/history")
def results_history(limit: int = 50) -> dict[str, Any]:
    return {"runs": db.list_runs(limit=int(limit))}


@router.get("/picks/latest")
def picks_latest() -> dict[str, Any]:
    run_id = db.latest_succeeded_run_id("run-monthly")
    if run_id is None:
        return {"run_id": None, "picks": []}
    picks = db.list_picks(run_id)
    return {"run_id": run_id, "picks": picks}


@router.get("/results/latest")
def results_latest() -> dict[str, Any]:
    run_id = db.latest_succeeded_run_id("run-monthly")
    if run_id is None:
        return {"run_id": None, "metrics": None}
    m = db.get_headline_metrics(run_id)
    return {"run_id": run_id, "metrics": m}

