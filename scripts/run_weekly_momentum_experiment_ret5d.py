"""
Weekly momentum experiment (Ret_5d) — 2023–2026 — isolated from monthly pipeline.

This addresses the "monthly-style signal on weekly clock" mismatch by using:
- predicted_alpha: Ret_5d (log return over last 5 trading days) from processed features
- actual_alpha: forward 5-trading-day log return computed from raw prices (diagnostic)

Rebalance grid:
- Every 5th NIFTY trading day from `data/processed/_NSEI_features.parquet`.

Backtest:
- Uses the existing `models.backtest.run_backtest_daily` with identical rules/costs.

Outputs:
- results/final_strategy/weekly_ret5d_ranking_predictions_2023_2026.csv
- results/final_strategy/backtest_results_weekly_ret5d_2023_2026.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.backtest import _load_price_series, _raw_ticker_stem, run_backtest_daily  # noqa: E402


def _load_nifty_trading_days(processed_dir: Path, start: str, end: str) -> np.ndarray:
    nifty_path = processed_dir / "_NSEI_features.parquet"
    if not nifty_path.is_file():
        raise SystemExit(f"Missing NIFTY features parquet: {nifty_path}")
    n = pd.read_parquet(nifty_path, columns=["Date"])
    n["Date"] = pd.to_datetime(n["Date"])
    nd = n["Date"].sort_values().unique()
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    nd = nd[(nd >= start_ts) & (nd <= end_ts)]
    if len(nd) == 0:
        raise SystemExit("No NIFTY trading days in requested window.")
    return nd


def _forward_log_return(prices: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    w = prices.loc[(prices.index >= d0) & (prices.index < d1)]
    if w.empty:
        return 0.0
    p0 = float(w.iloc[0])
    p1 = float(w.iloc[-1])
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0 or p1 <= 0:
        return 0.0
    return float(np.log(p1 / p0))


def build_weekly_ret5d_predictions(
    *,
    processed_dir: Path,
    raw_dir: Path,
    start: str,
    end: str,
    rebalance_every: int,
    out_csv: Path,
) -> pd.DataFrame:
    processed = processed_dir.resolve()
    raw_root = raw_dir.resolve()

    nd = _load_nifty_trading_days(processed, start, end)
    rebal = pd.to_datetime(nd[:: int(rebalance_every)])
    if len(rebal) < 2:
        raise SystemExit("Need at least 2 rebalance dates in range.")

    paths = sorted(
        p
        for p in processed.glob("*_features.parquet")
        if p.name != "_NSEI_features.parquet" and not p.stem.startswith("_")
    )
    if not paths:
        raise SystemExit(f"No stock feature files in {processed}")

    # Load panel for Ret_5d signal (fast) then compute forward 5d realized return (diagnostic).
    chunks: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p, columns=["Date", "Ticker", "Ret_5d"])
        chunks.append(df)
    panel = pd.concat(chunks, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.dropna(subset=["Ret_5d"])
    rebal_set = set(pd.Timestamp(x) for x in rebal)
    panel = panel[panel["Date"].isin(rebal_set)]

    # Price cache for forward returns
    price_cache: dict[str, pd.Series] = {}

    def get_prices(ticker: str) -> pd.Series:
        if ticker in price_cache:
            return price_cache[ticker]
        path = raw_root / f"{_raw_ticker_stem(ticker)}.parquet"
        price_cache[ticker] = _load_price_series(str(path))
        return price_cache[ticker]

    # Build a quick lookup for next rebalance date
    rebal_sorted = pd.DatetimeIndex(sorted(rebal))
    next_map = {rebal_sorted[i]: rebal_sorted[i + 1] for i in range(len(rebal_sorted) - 1)}

    actuals: list[float] = []
    for _, row in panel.iterrows():
        d0 = pd.Timestamp(row["Date"])
        d1 = pd.Timestamp(next_map.get(d0, d0))
        if d1 <= d0:
            actuals.append(0.0)
            continue
        t = str(row["Ticker"])
        px = get_prices(t)
        actuals.append(_forward_log_return(px, d0, d1))

    out = panel.rename(columns={"Ret_5d": "predicted_alpha"}).copy()
    out["actual_alpha"] = np.asarray(actuals, dtype=float)
    out["Period"] = "Test"
    out = out[["Date", "Ticker", "predicted_alpha", "actual_alpha", "Period"]].sort_values(
        ["Date", "predicted_alpha"], ascending=[True, False]
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2023-01-01")
    ap.add_argument("--end", type=str, default="2026-04-16")
    ap.add_argument("--rebalance-every", type=int, default=5)
    ap.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()

    out_pred = (
        ROOT
        / "results"
        / "final_strategy"
        / "weekly_ret5d_ranking_predictions_2023_2026.csv"
    )
    out_bt = (
        ROOT
        / "results"
        / "final_strategy"
        / "backtest_results_weekly_ret5d_2023_2026.csv"
    )

    preds = build_weekly_ret5d_predictions(
        processed_dir=Path(args.processed_dir),
        raw_dir=Path(args.raw_dir),
        start=str(args.start),
        end=str(args.end),
        rebalance_every=int(args.rebalance_every),
        out_csv=out_pred,
    )

    bt = run_backtest_daily(
        predictions_df=preds[["Date", "Ticker", "predicted_alpha", "actual_alpha"]],
        nifty_features_path=str(ROOT / "data/processed/_NSEI_features.parquet"),
        raw_dir=str(Path(args.raw_dir)),
        start=str(args.start),
        end=str(args.end),
        top_n=5,
        capital=100000,
        stop_loss=0.07,
        max_weight=0.25,
        portfolio_dd_cash_trigger=0.15,
        rebalance_friction_rate=0.002,
        turnover_penalty_score=0.0,
        kelly_p=0.5238,
        kelly_use_predicted_margin=True,
        kelly_scale_position=True,
        use_sector_cap=True,
        flat_regime_sizing=False,
    )
    out_bt.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out_bt, index=False)

    print(f"Wrote predictions: {out_pred}")
    print(f"Wrote backtest:    {out_bt}")


if __name__ == "__main__":
    main()

