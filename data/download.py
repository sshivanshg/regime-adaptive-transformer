"""
Download OHLCV history via yfinance into data/raw/{ticker}.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_ohlcv(
    ticker: str,
    start: str,
    end: str | None,
    out_dir: Path,
    interval: str = "1d",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No rows returned for {ticker} ({start=} {end=})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    out = out_dir / f"{ticker.upper()}.csv"
    df.to_csv(out)
    print(f"Wrote {len(df)} rows -> {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Download OHLCV with yfinance")
    p.add_argument("--tickers", nargs="+", default=["SPY"], help="Symbols, e.g. SPY QQQ")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None, help="Exclusive end date YYYY-MM-DD")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
    )
    p.add_argument("--interval", default="1d", choices=["1d", "1wk", "1mo"])
    args = p.parse_args()

    for t in args.tickers:
        download_ohlcv(t.strip(), args.start, args.end, args.out_dir, args.interval)


if __name__ == "__main__":
    main()
