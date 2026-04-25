from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeadlineMetrics:
    strategy_sharpe: float
    nifty_sharpe: float
    strategy_cagr: float
    nifty_cagr: float
    strategy_max_dd: float
    nifty_max_dd: float
    strategy_win_rate: float
    last_rebalance_date: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_sharpe": float(self.strategy_sharpe),
            "nifty_sharpe": float(self.nifty_sharpe),
            "strategy_cagr": float(self.strategy_cagr),
            "nifty_cagr": float(self.nifty_cagr),
            "strategy_max_dd": float(self.strategy_max_dd),
            "nifty_max_dd": float(self.nifty_max_dd),
            "strategy_win_rate": float(self.strategy_win_rate),
            "last_rebalance_date": str(self.last_rebalance_date),
        }


def compute_strategy_metrics(bt: pd.DataFrame, *, capital: float = 100_000) -> dict[str, float]:
    r = bt["portfolio_return"].dropna().astype(float)
    nav = bt["portfolio_value"].astype(float).values
    start_ts = pd.to_datetime(bt["date"].iloc[0])
    end_ts = pd.to_datetime(bt["date"].iloc[-1])
    span_years = (end_ts - start_ts).days / 365.25

    sharpe_net = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0
    total_ret = float(nav[-1] / float(capital) - 1.0)
    cagr = float((1 + total_ret) ** (1 / span_years) - 1) if span_years > 0 else 0.0

    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())
    win_rate = float((r > 0).mean())
    return {
        "sharpe_net": sharpe_net,
        "cagr": cagr,
        "max_dd": max_dd,
        "win_rate": win_rate,
    }


def compute_nifty_benchmark(nifty: pd.DataFrame, start: Any, end: Any, *, capital: float = 100_000) -> dict[str, float]:
    n = nifty.copy()
    n["Date"] = pd.to_datetime(n["Date"])
    n = n[(n["Date"] >= start) & (n["Date"] <= end)].sort_values("Date")
    px = n["Adj Close"].astype(float).values
    nav = capital * px / px[0]
    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())
    total_ret = float(nav[-1] / capital - 1.0)
    span_years = (n["Date"].iloc[-1] - n["Date"].iloc[0]).days / 365.25
    cagr = float((1 + total_ret) ** (1 / span_years) - 1) if span_years > 0 else 0.0
    daily_ret = pd.Series(px).pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    return {"cagr": cagr, "max_dd": max_dd, "sharpe": sharpe}


def load_backtest_csv(path: str | Path) -> pd.DataFrame:
    bt = pd.read_csv(path, parse_dates=["date"])
    if bt.empty:
        raise ValueError("Backtest CSV is empty")
    return bt


def load_nifty_raw(path: str | Path) -> pd.DataFrame:
    n = pd.read_parquet(path)
    if n.empty:
        raise ValueError("NIFTY parquet is empty")
    return n


def compute_headline_from_files(
    *,
    backtest_csv: str | Path,
    nifty_parquet: str | Path,
    capital: float = 100_000,
) -> HeadlineMetrics:
    bt = load_backtest_csv(backtest_csv)
    nifty = load_nifty_raw(nifty_parquet)
    strat = compute_strategy_metrics(bt, capital=capital)
    bench = compute_nifty_benchmark(nifty, bt["date"].iloc[0], bt["date"].iloc[-1], capital=capital)
    last_dt = pd.to_datetime(bt["date"].iloc[-1]).strftime("%Y-%m-%d")
    return HeadlineMetrics(
        strategy_sharpe=float(strat["sharpe_net"]),
        nifty_sharpe=float(bench["sharpe"]),
        strategy_cagr=float(strat["cagr"]),
        nifty_cagr=float(bench["cagr"]),
        strategy_max_dd=float(strat["max_dd"]),
        nifty_max_dd=float(bench["max_dd"]),
        strategy_win_rate=float(strat["win_rate"]),
        last_rebalance_date=last_dt,
    )

