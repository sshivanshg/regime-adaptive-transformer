"""
RAMT Data Pipeline — Step 1: Download
Downloads OHLCV for configured tickers (JPM, RELIANCE.NS, TCS.NS, HDFCBANK.NS,
EPIGRAL.NS) and benchmark indices (NIFTY50, S&P500). Default period 2010-01-01 to
2026-01-01; EPIGRAL.NS uses the last 10 years before END_DATE (see TICKER_DATE_OVERRIDES).
Saves raw CSVs to data/raw/.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import time

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# NIFTY 50 universe (Yahoo Finance tickers)
NIFTY_50_TICKERS: list[str] = [
    # IT
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    # Banking
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    # Financial
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "HDFC.NS",
    # Energy
    "RELIANCE.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    # Consumer
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "BRITANNIA.NS",
    # Auto
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    # Pharma
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    # Metals
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "JSWSTEEL.NS",
    "COALINDIA.NS",
    # Telecom
    "BHARTIARTL.NS",
    # Cement
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "SHREECEM.NS",
    # Others
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJAJ-AUTO.NS",
    "BPCL.NS",
    "DMART.NS",
    "INDUSINDBK.NS",
    "LT.NS",
    "M&M.NS",
    "TITAN.NS",
    "UPL.NS",
    "VEDL.NS",
    "WIPRO.NS",
    "ASIANPAINT.NS",
    # Benchmark
    "^NSEI",  # NIFTY 50 index
]

# NIFTY 200 universe
# To avoid hardcoding 200 symbols in code, we load from `data/nifty200_tickers.txt`.
# File format: one Yahoo ticker per line (e.g. RELIANCE.NS). Comments allowed with '#'.
NIFTY_200_TICKERS_FILE = Path(__file__).resolve().parent / "nifty200_tickers.txt"


def load_tickers_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ticker universe file: {path}\n"
            "Create it with one Yahoo ticker per line (e.g., RELIANCE.NS)."
        )
    warnings.warn(
        "Ticker universe file is treated as a static snapshot across all dates. "
        "If this file contains a 2026 NIFTY 200 list, earlier training periods remain "
        "survivorship-biased even though the model will not see membership changes as a "
        "time-varying feature.",
        RuntimeWarning,
        stacklevel=2,
    )
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


# Macro series (saved as data/raw/macro_{NAME}_raw.csv)
MACRO_TICKERS: dict[str, str] = {
    "USDINR": "USDINR=X",
    "CRUDE": "CL=F",
    "GOLD": "GC=F",
    "USVIX": "^VIX",
}

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

TICKERS: list[str] = load_tickers_from_file(NIFTY_200_TICKERS_FILE)
TICKER_DATE_OVERRIDES: dict[str, tuple[str, str]] = {}
RAW_DIR = Path(__file__).resolve().parent / "raw"
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# -----------------------------------------------------------------------------
# Column handling
# -----------------------------------------------------------------------------


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns that yfinance sometimes returns into a single level.

    Parameters
    ----------
    df : pd.DataFrame
        Raw frame from yfinance.

    Returns
    -------
    pd.DataFrame
        Copy with string column names suitable for selecting OHLCV fields.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip() for c in out.columns]
    return out


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------


def pearson_kurtosis(series: pd.Series) -> float:
    """
    Compute Pearson kurtosis (normal distribution has kurtosis 3).

    Parameters
    ----------
    series : pd.Series
        Sample values (e.g. log returns).

    Returns
    -------
    float
        Kurtosis, or NaN if undefined.
    """
    arr = series.dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < 4:
        return float("nan")
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=0))
    if sigma == 0.0:
        return float("nan")
    z = (arr - mu) / sigma
    return float(np.mean(z**4))


# -----------------------------------------------------------------------------
# Download and transform
# -----------------------------------------------------------------------------


def download_one_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV for one ticker, add log returns and ticker label.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol.
    start : str
        Inclusive start date (YYYY-MM-DD).
    end : str
        Exclusive end date (YYYY-MM-DD), per yfinance convention.

    Returns
    -------
    pd.DataFrame
        Columns include Date, OHLCV, Log_Return, Ticker; empty if download failed.
    """
    # yfinance can occasionally return empty data due to transient Yahoo throttling
    # or cookie/crumb issues. Use a small retry loop and fall back to Ticker.history.
    raw = pd.DataFrame()
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                actions=False,
                threads=False,
                progress=False,
            )
            if not raw.empty:
                break
            # fallback path
            raw = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )
            if not raw.empty:
                break
        except Exception as e:
            last_err = e
        # exponential-ish backoff
        time.sleep(1.0 + attempt * 2.0)

    if raw.empty:
        if last_err is not None:
            print(f"  [WARN] download failed for {ticker}: {last_err}")
        return pd.DataFrame()

    df = flatten_yfinance_columns(raw)
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[OHLCV_COLS].copy()
    df = df.reset_index()
    date_name = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_name: "Date"})

    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Ticker"] = ticker
    return df


def raw_csv_path(ticker: str, out_dir: Path) -> Path:
    """
    Path for raw CSV: dots in symbol replaced with underscores, suffix _raw.csv.

    Parameters
    ----------
    ticker : str
        Symbol string.
    out_dir : Path
        Directory for raw files.

    Returns
    -------
    Path
        Full output path.
    """
    safe = ticker.replace(".", "_").replace("^", "_")
    return out_dir / f"{safe}_raw.csv"


def raw_csv_path_for_stem(file_stem: str, out_dir: Path) -> Path:
    """
    Path for a raw CSV using an explicit filename stem (e.g. NIFTY50, SP500).

    Parameters
    ----------
    file_stem : str
        Safe stem without ``_raw.csv`` suffix.
    out_dir : Path
        Directory for raw files.

    Returns
    -------
    Path
        Full output path.
    """
    return out_dir / f"{file_stem}_raw.csv"


# -----------------------------------------------------------------------------
# Console output
# -----------------------------------------------------------------------------


def print_section_title(title: str) -> None:
    """Print a major section heading with separator lines."""
    line = "=" * 72
    print(f"\n{line}\n  {title}\n{line}")


def print_subsection(title: str) -> None:
    """Print a minor subsection divider."""
    print(f"\n  --- {title} ---")


def print_ticker_diagnostics(ticker: str, df: pd.DataFrame) -> None:
    """
    Print per-ticker diagnostics: shape, dtypes, NaNs, Close and log-return stats.

    Parameters
    ----------
    ticker : str
        Symbol label for the section header.
    df : pd.DataFrame
        Data from ``download_one_ticker``; may be empty.
    """
    print_section_title(f"Ticker: {ticker}")

    if df.empty:
        print("\n  [No data] yfinance returned an empty DataFrame — skipping save.")
        return

    start = pd.Timestamp(df["Date"].min()).date()
    end = pd.Timestamp(df["Date"].max()).date()
    print(f"\n  Date range     : {start}  →  {end}")
    print(f"  Shape (rows×cols): {df.shape[0]} × {df.shape[1]}")

    print_subsection("Dtypes")
    for col in df.columns:
        print(f"    {str(col):<14} {df[col].dtype}")

    print_subsection("NaN count (with % of rows)")
    n_rows = len(df)
    for col in df.columns:
        n_nan = int(df[col].isna().sum())
        pct = (n_nan / n_rows * 100.0) if n_rows else 0.0
        print(f"    {str(col):<14} {n_nan:>6}  ({pct:6.2f}%)")

    print_subsection("Close price")
    c = df["Close"]
    print(
        f"    min   = {c.min():.6f}\n"
        f"    max   = {c.max():.6f}\n"
        f"    mean  = {c.mean():.6f}\n"
        f"    std   = {c.std():.6f}"
    )

    lr = df["Log_Return"]
    lr_clean = lr.dropna()
    pk = pearson_kurtosis(lr)
    print_subsection("Log_Return")
    print(
        f"    mean      = {lr_clean.mean():.8f}\n"
        f"    std       = {lr_clean.std():.8f}\n"
        f"    skewness  = {lr_clean.skew():.6f}\n"
        f"    kurtosis  = {pk:.6f}  (Pearson; Gaussian = 3)\n"
        f"    min (worst day)  = {lr.min():.8f}\n"
        f"    max (best day)   = {lr.max():.8f}"
    )
    if not np.isnan(pk) and pk > 3.0:
        print(
            "\n  Note: Heavy tails confirmed — justifies regime-adaptive modeling."
        )


def print_cross_ticker_summary(rows: list[dict[str, object]]) -> None:
    """
    Print aligned cross-ticker summary table.

    Parameters
    ----------
    rows : list of dict
        Each dict has keys: Ticker, Rows, Start, End, Total_NaNs, LR_Kurtosis.
    """
    print_section_title("Cross-ticker summary")
    headers = (
        "Ticker",
        "Rows",
        "Start",
        "End",
        "Total NaNs",
        "Log Return Kurtosis",
    )
    col_w = [16, 8, 12, 12, 12, 22]
    head = (
        f"  {headers[0]:<{col_w[0]}}"
        f"{headers[1]:>{col_w[1]}}"
        f"  {headers[2]:<{col_w[2]}}"
        f"  {headers[3]:<{col_w[3]}}"
        f"  {headers[4]:>{col_w[4]}}"
        f"  {headers[5]:>{col_w[5]}}"
    )
    sep = "  " + "-" * (sum(col_w) + 4 * 2 + 6)
    print(f"\n{head}\n{sep}")
    for r in rows:
        t = str(r["Ticker"])
        rs = r["Rows"]
        s0 = r["Start"]
        e0 = r["End"]
        nn = r["Total_NaNs"]
        ku = r["LR_Kurtosis"]
        rs_s = "" if rs is None else str(rs)
        s_s = "" if s0 is None else str(s0)
        e_s = "" if e0 is None else str(e0)
        nn_s = "" if nn is None else str(nn)
        if ku is None or (isinstance(ku, float) and np.isnan(ku)):
            ku_s = "—"
        else:
            ku_s = f"{float(ku):.6f}"
        line = (
            f"  {t:<{col_w[0]}}"
            f"{rs_s:>{col_w[1]}}"
            f"  {s_s:<{col_w[2]}}"
            f"  {e_s:<{col_w[3]}}"
            f"  {nn_s:>{col_w[4]}}"
            f"  {ku_s:>{col_w[5]}}"
        )
        print(line)
    print()


def save_ticker_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Write dataframe to CSV, creating parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    """Download all configured tickers, print diagnostics, save CSVs, print summary."""
    print_section_title("RAMT — data download & first-look diagnostics")
    print(
        f"\n  Tickers  : {', '.join(TICKERS)}\n"
        f"  Macro    : {', '.join(f'{k}={v}' for k, v in MACRO_TICKERS.items())}\n"
        f"  Period   : {START_DATE}  →  {END_DATE}  (end exclusive)\n"
        f"  Output   : {RAW_DIR}"
    )

    summary_rows: list[dict[str, object]] = []

    for ticker in TICKERS:
        t0, t1 = TICKER_DATE_OVERRIDES.get(ticker, (START_DATE, END_DATE))
        df = download_one_ticker(ticker, t0, t1)
        print_ticker_diagnostics(ticker, df)

        if df.empty:
            summary_rows.append(
                {
                    "Ticker": ticker,
                    "Rows": None,
                    "Start": None,
                    "End": None,
                    "Total_NaNs": None,
                    "LR_Kurtosis": None,
                }
            )
            continue

        out_path = raw_csv_path(ticker, RAW_DIR)
        save_ticker_csv(df, out_path)
        print(f"\n  Saved → {out_path.resolve()}")

        total_nans = int(df.isna().sum().sum())
        pk = pearson_kurtosis(df["Log_Return"])
        summary_rows.append(
            {
                "Ticker": ticker,
                "Rows": len(df),
                "Start": pd.Timestamp(df["Date"].min()).date(),
                "End": pd.Timestamp(df["Date"].max()).date(),
                "Total_NaNs": total_nans,
                "LR_Kurtosis": pk,
            }
        )

    print_section_title("Macro series")
    for name, yahoo_sym in MACRO_TICKERS.items():
        df = download_one_ticker(yahoo_sym, START_DATE, END_DATE)
        label = f"{name} ({yahoo_sym})"
        print_ticker_diagnostics(label, df)

        if df.empty:
            summary_rows.append(
                {
                    "Ticker": f"macro_{name}",
                    "Rows": None,
                    "Start": None,
                    "End": None,
                    "Total_NaNs": None,
                    "LR_Kurtosis": None,
                }
            )
            continue

        out_path = raw_csv_path_for_stem(f"macro_{name}", RAW_DIR)
        save_ticker_csv(df, out_path)
        print(f"\n  Saved → {out_path.resolve()}")

        total_nans = int(df.isna().sum().sum())
        pk = pearson_kurtosis(df["Log_Return"])
        summary_rows.append(
            {
                "Ticker": f"macro_{name}",
                "Rows": len(df),
                "Start": pd.Timestamp(df["Date"].min()).date(),
                "End": pd.Timestamp(df["Date"].max()).date(),
                "Total_NaNs": total_nans,
                "LR_Kurtosis": pk,
            }
        )

    print_cross_ticker_summary(summary_rows)


if __name__ == "__main__":
    main()
