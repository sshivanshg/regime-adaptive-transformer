from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.src.paths import DB_PATH


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY,
              type TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,
              params_json TEXT NOT NULL,
              stdout_path TEXT,
              stderr_path TEXT,
              error_message TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              path TEXT NOT NULL,
              kind TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS headline_metrics (
              run_id TEXT PRIMARY KEY,
              strategy_sharpe REAL,
              nifty_sharpe REAL,
              strategy_cagr REAL,
              nifty_cagr REAL,
              strategy_max_dd REAL,
              nifty_max_dd REAL,
              strategy_win_rate REAL,
              last_rebalance_date TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS picks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              rebalance_date TEXT NOT NULL,
              ticker TEXT NOT NULL,
              weight REAL,
              regime TEXT,
              momentum REAL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )


def create_run(*, run_id: str, run_type: str, params: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (id, type, status, created_at, params_json)
            VALUES (?, ?, 'queued', ?, ?)
            """,
            (run_id, run_type, utc_now_iso(), json.dumps(params, sort_keys=True)),
        )


def update_run_status(
    run_id: str,
    *,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    error_message: str | None = None,
) -> None:
    cols: list[str] = ["status = ?"]
    vals: list[Any] = [status]
    if started_at is not None:
        cols.append("started_at = ?")
        vals.append(started_at)
    if finished_at is not None:
        cols.append("finished_at = ?")
        vals.append(finished_at)
    if stdout_path is not None:
        cols.append("stdout_path = ?")
        vals.append(stdout_path)
    if stderr_path is not None:
        cols.append("stderr_path = ?")
        vals.append(stderr_path)
    if error_message is not None:
        cols.append("error_message = ?")
        vals.append(error_message)

    vals.append(run_id)
    with connect() as conn:
        conn.execute(f"UPDATE runs SET {', '.join(cols)} WHERE id = ?", vals)


def get_run(run_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["params"] = json.loads(d.pop("params_json") or "{}")
        return d


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d.pop("params_json") or "{}")
            out.append(d)
        return out


def add_artifact(run_id: str, path: str, *, kind: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO artifacts (run_id, path, kind, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, path, kind, utc_now_iso()),
        )


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT path, kind, created_at FROM artifacts WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def upsert_headline_metrics(run_id: str, metrics: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO headline_metrics (
              run_id,
              strategy_sharpe,
              nifty_sharpe,
              strategy_cagr,
              nifty_cagr,
              strategy_max_dd,
              nifty_max_dd,
              strategy_win_rate,
              last_rebalance_date,
              created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              strategy_sharpe=excluded.strategy_sharpe,
              nifty_sharpe=excluded.nifty_sharpe,
              strategy_cagr=excluded.strategy_cagr,
              nifty_cagr=excluded.nifty_cagr,
              strategy_max_dd=excluded.strategy_max_dd,
              nifty_max_dd=excluded.nifty_max_dd,
              strategy_win_rate=excluded.strategy_win_rate,
              last_rebalance_date=excluded.last_rebalance_date
            """,
            (
                run_id,
                metrics.get("strategy_sharpe"),
                metrics.get("nifty_sharpe"),
                metrics.get("strategy_cagr"),
                metrics.get("nifty_cagr"),
                metrics.get("strategy_max_dd"),
                metrics.get("nifty_max_dd"),
                metrics.get("strategy_win_rate"),
                metrics.get("last_rebalance_date"),
                utc_now_iso(),
            ),
        )


def replace_picks(run_id: str, rebalance_date: str, picks: list[dict[str, Any]]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM picks WHERE run_id = ?", (run_id,))
        for p in picks:
            conn.execute(
                """
                INSERT INTO picks (run_id, rebalance_date, ticker, weight, regime, momentum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    rebalance_date,
                    p["ticker"],
                    p.get("weight"),
                    p.get("regime"),
                    p.get("momentum"),
                    utc_now_iso(),
                ),
            )


def latest_succeeded_run_id(run_type: str | None = None) -> str | None:
    with connect() as conn:
        if run_type:
            row = conn.execute(
                """
                SELECT id FROM runs
                WHERE status='succeeded' AND type=?
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (run_type,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id FROM runs
                WHERE status='succeeded'
                ORDER BY finished_at DESC
                LIMIT 1
                """
            ).fetchone()
        return str(row["id"]) if row else None


def get_headline_metrics(run_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM headline_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None


def list_picks(run_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT rebalance_date, ticker, weight, regime, momentum, created_at
            FROM picks
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

