"""
RAMT Data Pipeline — Step 2: Feature Engineering
Transforms raw OHLCV(+Adj Close) data into a feature matrix per ticker.

This version is refactored for the NIFTY 200 Parquet-based raw store:
- Batch-process all equity Parquet files in `data/raw/` (including ``_NSEI.parquet`` or ``_NSEI_raw.csv``).
- Compute returns (log) over 1d, 5d, 21d using **Adj Close**.
- Compute technicals: RSI(14), Bollinger Band distance, Volume Surge (Vol / SMA20 Vol).
- Merge macro series (INDIAVIX, CRUDE, USDINR, SP500) using 1-day lagged returns (no leakage).
- Target: Monthly_Alpha = ln(P_{t+21}/P_t) - ln(N_{t+21}/N_t), using **Adj Close**.
- Output: `data/processed/{ticker}_features.parquet` (no CSV).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from features.sectors import get_sector

# -----------------------------------------------------------------------------
# Paths & raw file mapping
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data" / "raw").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

START_DATE = "2020-01-01"
# yfinance end is exclusive; to include 2026-04-15 use 2026-04-16.
END_DATE_EXCLUSIVE = "2026-04-16"

# Benchmark used in Monthly_Alpha.
NIFTY_BENCHMARK_TICKER = "^NSEI"
NIFTY_BENCHMARK_PARQUET = RAW_DIR / "_NSEI.parquet"

# Macro series (names expected by the acquisition script).
MACRO_TICKERS: dict[str, str] = {
    "INDIAVIX": "^INDIAVIX",
    "CRUDE": "CL=F",
    "USDINR": "INR=X",
    "SP500": "^GSPC",
}
HMM_MIN_OBS = 60

# Required columns in raw parquet files created by scripts/fetch_nifty200.py
RAW_REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def list_stock_parquet_files(raw_dir: Path) -> list[Path]:
    """
    Return equity raw Parquet files to process (includes ``_NSEI.parquet``).

    Excludes macro series files (``macro_*.parquet``) only.
    """
    out: list[Path] = []
    for p in sorted(raw_dir.glob("*.parquet")):
        if p.name.startswith("macro_"):
            continue
        out.append(p)
    return out


def list_equity_input_paths(raw_dir: Path) -> list[Path]:
    """
    Parquet inputs under ``raw_dir`` plus ``_NSEI_raw.csv`` when no ``_NSEI.parquet`` exists.
    """
    paths = list_stock_parquet_files(raw_dir)
    nse_pq = raw_dir / "_NSEI.parquet"
    nse_csv = raw_dir / "_NSEI_raw.csv"
    if not nse_pq.exists() and nse_csv.exists():
        paths = [p for p in paths if p.name != "_NSEI_raw.csv"]
        paths.append(nse_csv)
    return sorted(set(paths))


def _calendar_from_benchmark(nifty_df: pd.DataFrame) -> pd.DatetimeIndex:
    cal = pd.to_datetime(nifty_df["Date"]).dt.tz_localize(None)
    cal = cal[(cal >= pd.Timestamp(START_DATE)) & (cal < pd.Timestamp(END_DATE_EXCLUSIVE))]
    return pd.DatetimeIndex(cal.unique()).sort_values()


def _align_equity_to_calendar(
    df: pd.DataFrame, calendar: pd.DatetimeIndex, ticker: str
) -> pd.DataFrame:
    """
    Align an equity OHLCV frame to the benchmark trading calendar.

    - If a stock IPOs / starts late (e.g. Feb 2020), we *pad the beginning* so it can still
      be used in training windows starting at START_DATE.
    - For padded rows: prices are set to the first available price (avoids zero/negative logs),
      volumes set to 0.
    - For internal gaps: forward-fill prices, keep volume at 0 for missing days.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out = out.sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date", drop=False)

    # Reindex to the full calendar (left join on stock index).
    out = out.reindex(calendar)
    out["Date"] = out.index
    out["Ticker"] = ticker

    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    for c in price_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    # Internal gaps: carry forward last known prices.
    out[price_cols] = out[price_cols].ffill()

    # Beginning padding: if still missing (stock starts after START_DATE), use first valid row.
    if out["Adj Close"].isna().any():
        first_valid = out["Adj Close"].first_valid_index()
        if first_valid is not None:
            first_row = out.loc[first_valid, price_cols]
            out.loc[:first_valid, price_cols] = out.loc[:first_valid, price_cols].fillna(first_row)

    # Volume: missing days → 0; no forward-fill for volume.
    out["Volume"] = out["Volume"].fillna(0.0)

    return out.reset_index(drop=True)


def _read_raw_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; have {list(df.columns)}")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _read_raw_equity_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"].astype(float)
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns and "Adj Close" in df.columns:
            df[col] = df["Adj Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    if "Ticker" not in df.columns:
        df.insert(0, "Ticker", NIFTY_BENCHMARK_TICKER)
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; have {list(df.columns)}")
    return df


def _read_raw_equity(path: Path) -> pd.DataFrame:
    """Load OHLCV (+Ticker) from Parquet or CSV (e.g. ``_NSEI_raw.csv``)."""
    suf = path.suffix.lower()
    if suf == ".csv":
        return _read_raw_equity_csv(path)
    if suf == ".parquet":
        return _read_raw_parquet(path)
    raise ValueError(f"Unsupported raw equity path: {path}")


# Full engineered schema per ticker (RAMT uses ``ALL_FEATURE_COLS`` subset in dataset.py).
FEATURE_OUTPUT_COLUMNS: list[str] = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Ret_1d",
    "Ret_5d",
    "Ret_21d",
    "Realized_Vol_20",
    "RSI_14",
    "BB_Dist",
    "Volume_Surge",
] + [f"Macro_{k}_Ret1d_L1" for k in MACRO_TICKERS.keys()] + [
    "Monthly_Alpha",
    "Sector_Alpha",
    "Daily_Return",
    "HMM_Regime",
    "Sector",
]


def build_features_table(
    df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    macro_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Full feature pipeline for one ticker OHLCV frame (same logic for index and stocks).
    """
    out = df.copy()
    out = add_returns_features(out)
    out = add_realized_vol_20(out)
    out = add_rsi_14(out)
    out = add_bollinger_distance(out)
    out = add_volume_surge(out)
    out = add_macro_lagged_returns(out, macro_data)
    out = compute_monthly_alpha_adjclose(out, nifty_df)
    out = add_daily_target(out)
    out = add_hmm_regime_full_history(out)

    tk = str(out["Ticker"].iloc[0]) if "Ticker" in out.columns and len(out) else ""
    out["Sector"] = get_sector(tk) if tk else "OTHER"
    out["Sector_Alpha"] = np.nan

    # Warm-up NaNs are expected from rolling indicators. Keep rows and use neutral defaults.
    fill_zero_cols = [
        "Ret_1d",
        "Ret_5d",
        "Ret_21d",
        "Realized_Vol_20",
        "RSI_14",
        "BB_Dist",
        "Volume_Surge",
        "Daily_Return",
        "HMM_Regime",
    ] + [f"Macro_{k}_Ret1d_L1" for k in MACRO_TICKERS.keys()]
    for c in fill_zero_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out = out[FEATURE_OUTPUT_COLUMNS]
    return out


def process_raw_equity_path(
    path: Path,
    nifty_df: pd.DataFrame,
    macro_data: dict[str, pd.DataFrame],
    processed_dir: Path,
) -> tuple[str, Path] | tuple[str, None]:
    """
    Read one raw file, compute features, write ``{{stem}}_features.parquet``.

    Returns ``(ticker_label, output_path)`` or ``(ticker_label, None)`` if empty after dropna.
    """
    try:
        df = _read_raw_equity(path)
        ticker = str(df["Ticker"].iloc[0]) if "Ticker" in df.columns and len(df) else path.stem

        # Align to benchmark calendar (inclusive START_DATE); pad early history if needed.
        cal = _calendar_from_benchmark(nifty_df)
        df = _align_equity_to_calendar(df, cal, ticker)

        n_orig = len(df)
        out = build_features_table(df, nifty_df, macro_data)
    except Exception as e:
        ticker = path.stem
        print(f"Skipping {ticker}: exception during feature build ({type(e).__name__}): {e}")
        return ticker, None

    # Keep rows even if some indicators are warm-start NaN; only require the label.
    if "Monthly_Alpha" in out.columns:
        out = out.dropna(subset=["Monthly_Alpha"])
    else:
        print(f"Skipping {ticker}: missing Monthly_Alpha column after feature build")
        return ticker, None

    if len(out) == 0:
        print(f"Skipping {ticker}: 0 rows after dropping rows without Monthly_Alpha (n_orig={n_orig})")
        return ticker, None

    out_path = processed_dir / f"{_safe_stem_from_ticker(ticker)}_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[{ticker}] rows: {n_orig} original → {len(out)} after dropna → saved {out_path.name}")
    return ticker, out_path


def _download_benchmark_if_missing() -> Path:
    if NIFTY_BENCHMARK_PARQUET.exists():
        return NIFTY_BENCHMARK_PARQUET

    csv_alt = NIFTY_BENCHMARK_PARQUET.parent / "_NSEI_raw.csv"
    if csv_alt.exists():
        return csv_alt

    print(f"Benchmark parquet missing; downloading {NIFTY_BENCHMARK_TICKER} …")
    data = yf.download(
        NIFTY_BENCHMARK_TICKER,
        start=START_DATE,
        end=END_DATE_EXCLUSIVE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        threads=False,
        progress=False,
    )
    data = _flatten_yfinance_columns(data)
    if data.empty:
        raise RuntimeError(f"Failed to download benchmark: {NIFTY_BENCHMARK_TICKER}")
    need = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    miss = [c for c in need if c not in data.columns]
    if miss:
        raise RuntimeError(f"Benchmark missing expected columns {miss}; have {list(data.columns)}")

    out = data[need].copy().reset_index()
    if "Date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out.insert(0, "Ticker", NIFTY_BENCHMARK_TICKER)
    NIFTY_BENCHMARK_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(NIFTY_BENCHMARK_PARQUET, index=False)
    return NIFTY_BENCHMARK_PARQUET


def load_macro_series(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load macro parquet files produced by scripts/fetch_nifty200.py and return
    dict keyed by canonical names in MACRO_TICKERS.
    """
    macro_paths = list(sorted(raw_dir.glob("macro_*.parquet")))
    if not macro_paths:
        raise FileNotFoundError(f"No macro parquet files found in: {raw_dir}")

    out: dict[str, pd.DataFrame] = {}
    for p in macro_paths:
        df = _read_raw_parquet(p)
        # Macro parquet includes "Ticker" column; keep it but compute features from Adj Close.
        # Heuristic mapping: prefer the first canonical name that appears in filename.
        p_upper = p.name.upper()
        matched = None
        for nm in MACRO_TICKERS.keys():
            if nm in p_upper:
                matched = nm
                break
        if matched is None:
            # fallback: take whatever after "macro_" until next "_" (still deterministic)
            matched = p.name.split("macro_", 1)[1].split("_", 1)[0].upper()

        out[matched] = df.set_index("Date", drop=False)

    missing = [k for k in MACRO_TICKERS.keys() if k not in out]
    if missing:
        raise FileNotFoundError(
            f"Missing required macro series {missing}. Found: {sorted(out.keys())}. "
            f"Expected files for {sorted(MACRO_TICKERS.keys())} in {raw_dir}."
        )
    return out


# -----------------------------------------------------------------------------
# Feature computation (Adj Close based)
# -----------------------------------------------------------------------------


def add_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    px = out["Adj Close"].astype(float).replace(0.0, np.nan)
    r1 = px / px.shift(1)
    r5 = px / px.shift(5)
    r21 = px / px.shift(21)
    out["Ret_1d"] = np.log(r1.where(r1 > 0.0))
    out["Ret_5d"] = np.log(r5.where(r5 > 0.0))
    out["Ret_21d"] = np.log(r21.where(r21 > 0.0))
    return out


def add_realized_vol_20(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Realized_Vol_20"] = out["Ret_1d"].rolling(20).std()
    return out


def add_daily_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tactical head target (sanity-check): next-day return using Adj Close log-return.
    """
    out = df.copy()
    out["Daily_Return"] = out["Ret_1d"].shift(-1)
    return out


def _semantic_hmm_mapping(mean_by_state: dict[int, float]) -> dict[int, int]:
    """
    Map raw 3-state HMM states to semantic regime codes:
    bull=1 (highest mean return), bear=2 (lowest mean return), high_vol=0 (middle).
    """
    active = {int(k): float(v) for k, v in mean_by_state.items() if np.isfinite(v)}
    if not active:
        raise ValueError("Cannot map semantic HMM states without any active states.")

    states = sorted(active.keys(), key=lambda s: active[s])
    if len(states) == 1:
        return {states[0]: 0}
    if len(states) == 2:
        low, high = states[0], states[1]
        return {high: 1, low: 2}

    low = states[0]
    high = states[-1]
    middle_states = states[1:-1]
    mid = middle_states[len(middle_states) // 2]

    mapping = {high: 1, low: 2}
    for s in middle_states:
        mapping[s] = 0
    mapping[mid] = 0
    return mapping


def _build_gaussian_hmm(prev_hmm: GaussianHMM | None = None) -> GaussianHMM:
    """
    Create a GaussianHMM, warm-starting from the previous expanding-window fit when available.
    """
    init_params = "stmc" if prev_hmm is None else ""
    hmm = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=300,
        tol=1e-4,
        random_state=42,
        verbose=False,
        init_params=init_params,
        params="stmc",
    )
    if prev_hmm is not None:
        hmm.startprob_ = prev_hmm.startprob_.copy()
        hmm.transmat_ = prev_hmm.transmat_.copy()
        hmm.means_ = prev_hmm.means_.copy()
        hmm._covars_ = prev_hmm._covars_.copy()
    return hmm


def _fit_hmm_silently(hmm: GaussianHMM, X: np.ndarray) -> GaussianHMM:
    """
    hmmlearn occasionally writes non-convergence notes directly to stderr/stdout even
    with ``verbose=False``. Keep feature generation logs readable.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        hmm.fit(X)
    return hmm


def add_hmm_regime_full_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a 3-state Gaussian HMM on an expanding window so the regime for each date
    only uses information available up to that date.

    Uses [Ret_1d, Realized_Vol_20] to capture high-vol as a distinct state.
    """
    out = df.copy()
    lr = out["Ret_1d"]
    rv20 = out["Realized_Vol_20"]
    mask = lr.notna() & rv20.notna()
    # Default regime is Neutral/High-Vol (0) so we never drop a stock solely due to HMM issues.
    out["HMM_Regime"] = 0.0

    if int(mask.sum()) < HMM_MIN_OBS:
        return out

    idx = out.index[mask]
    lr_vals = lr[mask].to_numpy(dtype=float)
    rv_vals = rv20[mask].to_numpy(dtype=float)
    X_raw = np.column_stack([lr_vals, rv_vals])

    prev_hmm: GaussianHMM | None = None
    prev_regime = float("nan")
    fit_failures = 0

    for end_pos in range(HMM_MIN_OBS - 1, len(idx)):
        X_hist_raw = X_raw[: end_pos + 1]
        mu = X_hist_raw.mean(axis=0)
        sigma = X_hist_raw.std(axis=0)
        sigma = np.where(sigma == 0.0, 1.0, sigma)
        X_hist = (X_hist_raw - mu) / sigma

        try:
            hmm = _build_gaussian_hmm(prev_hmm)
            hmm = _fit_hmm_silently(hmm, X_hist)
            raw_states = hmm.predict(X_hist)

            means: dict[int, float] = {}
            lr_hist = lr_vals[: end_pos + 1]
            for k in (0, 1, 2):
                sel = raw_states == k
                means[k] = float(np.mean(lr_hist[sel])) if np.any(sel) else float("nan")

            mapping = _semantic_hmm_mapping(means)
            prev_regime = float(mapping[int(raw_states[-1])])
            prev_hmm = hmm
        except Exception:
            fit_failures += 1
            # If we can't fit at all yet, fall back to Neutral/High-Vol (0).
            if not np.isfinite(prev_regime):
                prev_regime = 0.0

        out.loc[idx[end_pos], "HMM_Regime"] = prev_regime

    if fit_failures:
        warnings.warn(
            f"Causal HMM fit fell back to the prior regime {fit_failures} time(s). "
            "Inspect data quality if this becomes frequent.",
            RuntimeWarning,
            stacklevel=2,
        )
    out["HMM_Regime"] = out["HMM_Regime"].astype("float")
    return out


def add_rsi_14(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Adj Close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / 14.0
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    out["RSI_14"] = rsi
    return out


def add_bollinger_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger distance: standardized distance from 20d moving average in 2-sigma units.
    """
    out = df.copy()
    px = out["Adj Close"].astype(float)
    ma20 = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    denom = (2.0 * std20).replace(0.0, np.nan)
    out["BB_Dist"] = (px - ma20) / denom
    return out


def add_volume_surge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol = out["Volume"].astype(float)
    sma20 = vol.rolling(20).mean().replace(0.0, np.nan)
    out["Volume_Surge"] = vol / sma20
    return out


def add_macro_lagged_returns(df: pd.DataFrame, macro: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge macro 1d log returns with 1-day lag to avoid leakage.
    """
    out = df.copy()
    out = out.set_index("Date", drop=False)

    for name in MACRO_TICKERS.keys():
        mdf = macro[name]
        px = mdf["Adj Close"].astype(float).replace(0.0, np.nan)
        r1 = px / px.shift(1)
        mret = np.log(r1.where(r1 > 0.0)).shift(1)  # lag by 1 trading day
        mret = mret.rename(f"Macro_{name}_Ret1d_L1")

        # Left-join onto the stock calendar, then forward-fill to avoid dropping rows
        # from minor macro data gaps. Any remaining NaNs (e.g. very beginning) → 0.
        out = out.join(mret, how="left")
        out[mret.name] = out[mret.name].ffill().fillna(0.0)

    return out.reset_index(drop=True)


def compute_monthly_alpha_adjclose(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly_Alpha = ln(P_{t+21}/P_t) - ln(N_{t+21}/N_t), using Adj Close.
    """
    out = df.copy()
    out = out.set_index("Date", drop=False)
    nifty = nifty.set_index("Date", drop=False)

    p = out["Adj Close"].astype(float).replace(0.0, np.nan)
    r_fwd = p.shift(-21) / p
    stock_fwd = np.log(r_fwd.where(r_fwd > 0.0))

    aligned = out.join(nifty[["Adj Close"]].rename(columns={"Adj Close": "NIFTY_AdjClose"}), how="left")
    n = aligned["NIFTY_AdjClose"].astype(float).replace(0.0, np.nan)
    n_fwd = n.shift(-21) / n
    nifty_fwd = np.log(n_fwd.where(n_fwd > 0.0))

    out["Monthly_Alpha"] = stock_fwd - nifty_fwd
    return out.reset_index(drop=True)


def apply_sector_alpha_panel(processed_dir: Path) -> None:
    """
    Cross-sectional step: for each (Date, Sector) cohort, demean Monthly_Alpha by
    sector median so Sector_Alpha measures intra-sector relative strength.

    Must run after all per-ticker parquets exist. Idempotent if Sector_Alpha
    already populated.
    """
    paths = sorted(processed_dir.glob("*_features.parquet"))
    # Skip benchmark / non-equity feature files if any
    paths = [p for p in paths if not p.name.startswith("_")]
    if not paths:
        return

    chunks: list[pd.DataFrame] = []
    for p in paths:
        raw = pd.read_parquet(p)
        if "Sector" not in raw.columns:
            raw["Sector"] = raw["Ticker"].map(lambda x: get_sector(str(x)))
        chunks.append(raw[["Date", "Ticker", "Monthly_Alpha", "Sector"]])
    panel = pd.concat(chunks, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    if panel["Sector"].isna().any():
        panel["Sector"] = panel["Ticker"].map(lambda x: get_sector(str(x)))
    med = panel.groupby(["Date", "Sector"], sort=False)["Monthly_Alpha"].transform("median")
    panel["Sector_Alpha"] = panel["Monthly_Alpha"] - med

    for p in paths:
        df = pd.read_parquet(p)
        df["Date"] = pd.to_datetime(df["Date"])
        t = str(df["Ticker"].iloc[0])
        sub = panel.loc[panel["Ticker"] == t, ["Date", "Sector_Alpha"]]
        df = df.drop(columns=["Sector_Alpha"], errors="ignore").merge(
            sub, on="Date", how="left"
        )
        df.to_parquet(p, index=False)

    print(f"apply_sector_alpha_panel: wrote Sector_Alpha for {len(paths)} feature files.")


def _safe_stem_from_ticker(ticker: str) -> str:
    return (
        str(ticker)
        .replace(".", "_")
        .replace("^", "_")
        .replace("=", "_")
        .replace("/", "_")
        .replace("&", "_")
        .replace("-", "_")
        .rstrip("_")
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    bench_path = _download_benchmark_if_missing()
    equity_paths = list_equity_input_paths(RAW_DIR)
    if not equity_paths:
        raise FileNotFoundError(f"No equity raw files found in: {RAW_DIR}")

    nifty_df = _read_raw_equity(bench_path)
    macro_data = load_macro_series(RAW_DIR)

    for p in equity_paths:
        try:
            process_raw_equity_path(p, nifty_df, macro_data, PROCESSED_DIR)
        except Exception as e:
            print(f"Skipping {p.stem}: unhandled exception ({type(e).__name__}): {e}")

    apply_sector_alpha_panel(PROCESSED_DIR)


if __name__ == "__main__":
    main()
