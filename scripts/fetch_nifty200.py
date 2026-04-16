"""
Fetch NIFTY 200 universe + macro indicators and save as Parquet.

Requirements implemented:
- Pull official NIFTY 200 constituents CSV from NSE archives.
- Convert equity symbols to Yahoo Finance tickers by appending ".NS".
- Download daily OHLCV + Adj Close from 2020-01-01 to 2026-04-15 (inclusive end handled).
- Also download: ^INDIAVIX, CL=F, INR=X, ^GSPC.
- Rate limit: sleep(0.5) between tickers.
- Storage: write one parquet per series under data/raw/ (no 200 CSVs).
- Missing history: save whatever is available (no filling).
- Validation: print totals, failures, and disk usage of created parquet files.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

NSE_NIFTY200_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

START_DATE = "2020-01-01"
# User requirement is inclusive through 2026-04-15; yfinance `end` is exclusive.
END_DATE_EXCLUSIVE = "2026-04-16"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

MACRO_TICKERS: dict[str, str] = {
    "INDIAVIX": "^INDIAVIX",
    "CRUDE": "CL=F",
    "USDINR": "INR=X",
    "SP500": "^GSPC",
}

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _safe_stem(ticker: str) -> str:
    # Keep filenames portable; avoid special chars from indices and FX tickers.
    return (
        ticker.replace(".", "_")
        .replace("^", "_")
        .replace("=", "_")
        .replace("/", "_")
        .replace("&", "_")
        .replace("-", "_")
        .strip("_")
    )


def _download_daily_ohlcv_adjclose(ticker: str, start: str, end_exclusive: str) -> pd.DataFrame:
    """
    Download daily OHLCV + Adj Close using yfinance with a small retry loop.
    Returns empty df on failure or empty data.
    """
    raw = pd.DataFrame()
    last_err: Exception | None = None

    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,  # keep "Adj Close" explicitly
                actions=False,
                threads=False,
                progress=False,
            )
            if not raw.empty:
                break

            # Fallback sometimes succeeds when download() is throttled
            raw = yf.Ticker(ticker).history(
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            if not raw.empty:
                break
        except Exception as e:
            last_err = e

        time.sleep(1.0 + attempt * 2.0)

    if raw.empty:
        if last_err is not None:
            raise RuntimeError(f"yfinance failed for {ticker}: {last_err}") from last_err
        raise RuntimeError(f"yfinance returned empty data for {ticker}")

    df = _flatten_yfinance_columns(raw)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df = df.reset_index()
    # yfinance uses "Date" or "Datetime" depending on interval/asset; normalize.
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns:
        # last resort: first column name is the index after reset
        df = df.rename(columns={df.columns[0]: "Date"})

    # Enforce stable dtypes for parquet
    df["Date"] = pd.to_datetime(df["Date"], utc=False).dt.tz_localize(None)
    df.insert(0, "Ticker", ticker)
    return df


def fetch_nifty200_symbols() -> list[str]:
    """
    Fetch official NIFTY 200 list from NSE and return Yahoo tickers like RELIANCE.NS.
    """
    warnings.warn(
        "Fetching the current NIFTY 200 constituent snapshot (April 2026 context). "
        "This is survivorship-biased for pre-2026 backtests because historical index "
        "membership is not reconstructed here.",
        RuntimeWarning,
        stacklevel=2,
    )
    table = pd.read_csv(NSE_NIFTY200_CSV_URL)
    if "Symbol" not in table.columns:
        raise RuntimeError(f"NSE CSV missing Symbol column. Columns: {list(table.columns)}")

    symbols = (
        table["Symbol"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA})
        .dropna()
        .unique()
        .tolist()
    )
    # Ensure deterministic ordering
    symbols = sorted(symbols, key=str)
    return [f"{s}.NS" if not str(s).endswith(".NS") else str(s) for s in symbols]


@dataclass(frozen=True)
class SaveResult:
    ticker: str
    ok: bool
    rows: int | None = None
    out_path: Path | None = None
    err: str | None = None


def _write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _total_size_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"


def main() -> None:
    print("Fetching NIFTY 200 tickers from NSE…")
    tickers = fetch_nifty200_symbols()
    print(f"Found {len(tickers)} tickers.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results: list[SaveResult] = []
    parquet_paths: list[Path] = []

    print(f"\nDownloading equities (rate-limited @ 0.5s) {START_DATE} → 2026-04-15 …")
    for i, ticker in enumerate(tickers, start=1):
        try:
            df = _download_daily_ohlcv_adjclose(ticker, START_DATE, END_DATE_EXCLUSIVE)
            out_path = RAW_DIR / f"{_safe_stem(ticker)}.parquet"
            _write_parquet(df, out_path)
            results.append(SaveResult(ticker=ticker, ok=True, rows=len(df), out_path=out_path))
            parquet_paths.append(out_path)
            print(f"[{i:03d}/{len(tickers):03d}] OK   {ticker:<16} rows={len(df):5d} → {out_path.name}")
        except Exception as e:
            results.append(SaveResult(ticker=ticker, ok=False, err=str(e)))
            print(f"[{i:03d}/{len(tickers):03d}] FAIL {ticker:<16} {e}")
        time.sleep(0.5)

    print("\nDownloading macro series…")
    for name, mticker in MACRO_TICKERS.items():
        try:
            df = _download_daily_ohlcv_adjclose(mticker, START_DATE, END_DATE_EXCLUSIVE)
            out_path = RAW_DIR / f"macro_{_safe_stem(name)}_{_safe_stem(mticker)}.parquet"
            _write_parquet(df, out_path)
            results.append(SaveResult(ticker=mticker, ok=True, rows=len(df), out_path=out_path))
            parquet_paths.append(out_path)
            print(f"OK   {name:<8} {mticker:<10} rows={len(df):5d} → {out_path.name}")
        except Exception as e:
            results.append(SaveResult(ticker=mticker, ok=False, err=str(e)))
            print(f"FAIL {name:<8} {mticker:<10} {e}")
        time.sleep(0.5)

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    total_bytes = _total_size_bytes(parquet_paths)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Total equities requested : {len(tickers)}")
    print(f"Total series saved       : {len(ok)} (includes macro)")
    print(f"Failed series            : {len(failed)}")
    print(f"Total Parquet disk usage : {_human_bytes(total_bytes)}  ({len(parquet_paths)} files)")
    if failed:
        print("\nFailed tickers:")
        for r in failed:
            print(f"- {r.ticker}: {r.err}")


if __name__ == "__main__":
    main()
