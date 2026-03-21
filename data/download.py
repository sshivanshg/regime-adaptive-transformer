"""
RAMT (Regime-Adaptive Multimodal Transformer) — equity data download and diagnostics.

Run: python data/download.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TICKERS: list[str] = ["JPM", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
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
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
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
    safe = ticker.replace(".", "_")
    return out_dir / f"{safe}_raw.csv"


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
        f"\n  Tickers : {', '.join(TICKERS)}\n"
        f"  Period  : {START_DATE}  →  {END_DATE}  (end exclusive)\n"
        f"  Output  : {RAW_DIR}"
    )

    summary_rows: list[dict[str, object]] = []

    for ticker in TICKERS:
        df = download_one_ticker(ticker, START_DATE, END_DATE)
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

    print_cross_ticker_summary(summary_rows)


if __name__ == "__main__":
    main()
