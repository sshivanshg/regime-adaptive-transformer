"""
Download NSE **large-cap index** constituents (official CSVs) + yfinance daily OHLCV.

**Universe (pick one):**

| Index      | ~# | Meaning |
|------------|---|--------|
| ``nifty50``  | 50 | NIFTY 50 — largest/most liquid on NSE |
| ``nifty100`` | 100 | NIFTY 100 |
| ``nifty200`` | 200 | NIFTY 200 (default) |
| ``nifty500`` | 500 | NIFTY 500 — broad large/mid cap |

Use ``--max-symbols 400`` to cap (e.g. top of the CSV order for ``nifty500``). CSVs are **today’s**
NSE snapshot; true historical membership needs dated constituent files from NSE.

Also downloads: ``^NSEI`` benchmark, ``^INDIAVIX``, ``CL=F``, ``INR=X``, ``^GSPC``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

# Official NSE index constituent CSVs (Symbol column). Same host as existing NIFTY 200 pipeline.
NSE_INDEX_CSV_URL: dict[str, str] = {
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}

NSE_NIFTY200_CSV_URL = NSE_INDEX_CSV_URL["nifty200"]

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


def load_universe_file(path: Path) -> list[str]:
    """
    One Yahoo equity ticker per line (``RELIANCE.NS``). Lines starting with ``#`` are comments.
    Bare ``SYMBOL`` (no suffix) is treated as ``SYMBOL.NS``.
    """
    text = Path(path).expanduser().resolve().read_text(encoding="utf-8")
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        t = line if line.endswith(".NS") else f"{line}.NS"
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fetch_nse_index_yahoo_tickers(
    csv_url: str,
    *,
    max_symbols: int | None = None,
) -> list[str]:
    """
    Load NSE index CSV, return Yahoo symbols ``SYMBOL.NS`` in **CSV row order** (deduped).
    """
    table = pd.read_csv(csv_url)
    if "Symbol" not in table.columns:
        raise RuntimeError(f"NSE CSV missing Symbol column. Columns: {list(table.columns)}")

    out: list[str] = []
    seen: set[str] = set()
    for raw in table["Symbol"].astype(str).str.strip():
        if raw in ("", "nan", "NaN") or raw in seen:
            continue
        seen.add(raw)
        y = f"{raw}.NS" if not raw.endswith(".NS") else raw
        out.append(y)
        if max_symbols is not None and len(out) >= int(max_symbols):
            break
    return out


def fetch_nifty200_symbols() -> list[str]:
    """
    Fetch official NIFTY 200 list from NSE and return Yahoo tickers like RELIANCE.NS.

    Order is **sorted** for backward compatibility with older scripts.
    """
    warnings.warn(
        "Fetching the current NIFTY 200 constituent snapshot. "
        "Survivorship-biased for historical backtests unless you use period-specific NSE files.",
        RuntimeWarning,
        stacklevel=2,
    )
    tickers = fetch_nse_index_yahoo_tickers(NSE_NIFTY200_CSV_URL)
    return sorted(tickers, key=str)


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


NIFTY_BENCHMARK_YAHOO = "^NSEI"


def download_universe(
    start: str,
    end_exclusive: str,
    raw_dir: Path,
    *,
    tickers: list[str] | None = None,
    include_macros: bool = True,
    sleep_s: float = 0.5,
    skip_benchmark: bool = False,
) -> list[SaveResult]:
    """
    Download NIFTY benchmark (``_NSEI.parquet``), equity constituents, and optional macro series.

    ``end_exclusive`` follows yfinance (exclusive end date).
    """
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        print("Fetching NIFTY 200 tickers from NSE (sorted, legacy)…")
        tickers = fetch_nifty200_symbols()
        print(f"Found {len(tickers)} tickers.")

    results: list[SaveResult] = []
    parquet_paths: list[Path] = []

    if not skip_benchmark:
        bench_out = raw_dir / "_NSEI.parquet"
        if bench_out.exists():
            print(f"Using existing benchmark: {bench_out.name}")
            results.append(
                SaveResult(ticker=NIFTY_BENCHMARK_YAHOO, ok=True, rows=None, out_path=bench_out)
            )
        else:
            print(f"Downloading NIFTY benchmark {NIFTY_BENCHMARK_YAHOO} …")
            try:
                df = _download_daily_ohlcv_adjclose(NIFTY_BENCHMARK_YAHOO, start, end_exclusive)
                _write_parquet(df, bench_out)
                results.append(
                    SaveResult(
                        ticker=NIFTY_BENCHMARK_YAHOO,
                        ok=True,
                        rows=len(df),
                        out_path=bench_out,
                    )
                )
                parquet_paths.append(bench_out)
                print(f"OK   {NIFTY_BENCHMARK_YAHOO} rows={len(df):5d} → {bench_out.name}")
            except Exception as e:
                results.append(SaveResult(ticker=NIFTY_BENCHMARK_YAHOO, ok=False, err=str(e)))
                print(f"FAIL {NIFTY_BENCHMARK_YAHOO} {e}")
            time.sleep(sleep_s)

    print(f"\nDownloading equities (rate-limited @ {sleep_s}s) {start} → {end_exclusive} …")
    for i, ticker in enumerate(tickers, start=1):
        try:
            df = _download_daily_ohlcv_adjclose(ticker, start, end_exclusive)
            out_path = raw_dir / f"{_safe_stem(ticker)}.parquet"
            _write_parquet(df, out_path)
            results.append(SaveResult(ticker=ticker, ok=True, rows=len(df), out_path=out_path))
            parquet_paths.append(out_path)
            print(f"[{i:03d}/{len(tickers):03d}] OK   {ticker:<16} rows={len(df):5d} → {out_path.name}")
        except Exception as e:
            results.append(SaveResult(ticker=ticker, ok=False, err=str(e)))
            print(f"[{i:03d}/{len(tickers):03d}] FAIL {ticker:<16} {e}")
        time.sleep(sleep_s)

    if include_macros:
        print("\nDownloading macro series…")
        for name, mticker in MACRO_TICKERS.items():
            try:
                df = _download_daily_ohlcv_adjclose(mticker, start, end_exclusive)
                out_path = raw_dir / f"macro_{_safe_stem(name)}_{_safe_stem(mticker)}.parquet"
                _write_parquet(df, out_path)
                results.append(SaveResult(ticker=mticker, ok=True, rows=len(df), out_path=out_path))
                parquet_paths.append(out_path)
                print(f"OK   {name:<8} {mticker:<10} rows={len(df):5d} → {out_path.name}")
            except Exception as e:
                results.append(SaveResult(ticker=mticker, ok=False, err=str(e)))
                print(f"FAIL {name:<8} {mticker:<10} {e}")
            time.sleep(sleep_s)

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    total_bytes = _total_size_bytes(parquet_paths)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Raw directory            : {raw_dir}")
    print(f"Total equities requested : {len(tickers)}")
    print(f"Total series saved       : {len(ok)} (includes benchmark + macro if enabled)")
    print(f"Failed series            : {len(failed)}")
    print(f"Total Parquet disk usage : {_human_bytes(total_bytes)}  ({len(parquet_paths)} files)")
    if failed:
        print("\nFailed tickers:")
        for r in failed:
            print(f"- {r.ticker}: {r.err}")

    eq_set = set(tickers)
    eq_ok = [r for r in ok if r.ticker in eq_set]
    eq_fail = [r for r in failed if r.ticker in eq_set]
    stats = {
        "equities_requested": len(tickers),
        "equities_downloaded_ok": len(eq_ok),
        "equities_failed": len(eq_fail),
        "failed_equity_tickers": [r.ticker for r in eq_fail],
        "failed_equity_errors": {r.ticker: (r.err or "") for r in eq_fail},
    }
    stats_path = raw_dir / "_fetch_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nWrote {stats_path.name} (equity OK {len(eq_ok)} / requested {len(tickers)})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NSE index constituents (NIFTY 50/100/200/500) + macros via yfinance"
    )
    parser.add_argument("--start", default=START_DATE, help="Inclusive start (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        default=END_DATE_EXCLUSIVE,
        help="Exclusive end for yfinance (YYYY-MM-DD); last day included is the day before",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--index",
        choices=sorted(NSE_INDEX_CSV_URL.keys()),
        default="nifty200",
        help="NSE index CSV to use (default: nifty200). Use nifty500 for up to 500 names.",
    )
    parser.add_argument(
        "--nse-csv-url",
        type=str,
        default=None,
        help="Override: full URL to an NSE-style CSV with a Symbol column",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Cap number of equities after CSV order (e.g. 400 with --index nifty500)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print symbol count and path, save list if --save-symbol-list set, then exit",
    )
    parser.add_argument(
        "--save-symbol-list",
        type=Path,
        default=None,
        help="Write one Yahoo ticker per line (e.g. symbols.txt for inspection)",
    )
    parser.add_argument(
        "--no-macros",
        action="store_true",
        help="Skip INDIAVIX, crude, USDINR, SP500",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between tickers")
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="YAML-free list: one Yahoo ticker per line (overrides --index / --nse-csv-url)",
    )
    args = parser.parse_args()

    if args.universe_file is not None:
        p = Path(args.universe_file).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--universe-file not found: {p}")
        tickers = load_universe_file(p)
        csv_url = f"file:{p}"
    else:
        csv_url = args.nse_csv_url or NSE_INDEX_CSV_URL[args.index]
        tickers = fetch_nse_index_yahoo_tickers(csv_url, max_symbols=args.max_symbols)

    if args.list_only:
        src = "universe-file" if args.universe_file is not None else (
            "custom" if args.nse_csv_url else args.index
        )
        print(f"Source       : {src}")
        print(f"CSV / file   : {csv_url}")
        print(f"Symbol count : {len(tickers)}")
        if args.save_symbol_list is not None:
            args.save_symbol_list.parent.mkdir(parents=True, exist_ok=True)
            args.save_symbol_list.write_text("\n".join(tickers) + "\n", encoding="utf-8")
            print(f"Wrote list   : {args.save_symbol_list.resolve()}")
        else:
            print("First 15 tickers:", ", ".join(tickers[:15]))
        return

    if args.universe_file is None:
        warnings.warn(
            "Using current NSE constituent snapshot; historical backtests are survivorship-biased "
            "unless you supply period-specific NSE CSVs via --nse-csv-url or --universe-file.",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        warnings.warn(
            "Using --universe-file: you are responsible for point-in-time membership vs survivorship bias.",
            RuntimeWarning,
            stacklevel=1,
        )

    download_universe(
        args.start,
        args.end,
        args.raw_dir,
        tickers=tickers,
        include_macros=not args.no_macros,
        sleep_s=args.sleep,
    )


if __name__ == "__main__":
    main()
