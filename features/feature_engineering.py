"""
RAMT Data Pipeline — Step 2: Feature Engineering
Transforms raw OHLCV data into a 36-feature matrix per ticker.
Feature groups: lagged returns, volatility, technical indicators,
momentum, volume, HMM regime labels, cross-asset correlation.
Saves processed CSVs to data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# -----------------------------------------------------------------------------
# Paths & raw file mapping
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data" / "raw").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILES = [
    "JPM_raw.csv",
    "RELIANCE_NS_raw.csv",
    "TCS_NS_raw.csv",
    "HDFCBANK_NS_raw.csv",
    "EPIGRAL_NS_raw.csv",
]

LAG_DAYS = [1, 2, 3, 5, 10, 20]

# Indian listings use NIFTY; US (JPM) uses S&P 500
INDIAN_TICKER_SUBSTR = ".NS"


# -----------------------------------------------------------------------------
# Group 1 — Lagged Returns
# -----------------------------------------------------------------------------


def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged log-return columns so the model can use recent return history.

    For each n in {1,2,3,5,10,20}, Return_Lag_n is Log_Return shifted by n trading
    days (memory of past shocks).
    """
    out = df.copy()
    for n in LAG_DAYS:
        out[f"Return_Lag_{n}"] = out["Log_Return"].shift(n)
    return out


# -----------------------------------------------------------------------------
# Group 2 — Volatility Features
# -----------------------------------------------------------------------------


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add realized volatility horizons, Garman–Klass range-based volatility, and
    short/long vol ratio to capture volatility level and regime shifts.
    """
    out = df.copy()
    lr = out["Log_Return"]
    out["Realized_Vol_5"] = lr.rolling(5).std()
    out["Realized_Vol_20"] = lr.rolling(20).std()
    out["Realized_Vol_60"] = lr.rolling(60).std()

    hl = np.log(out["High"] / out["Low"])
    co = np.log(out["Close"] / out["Open"])
    out["Garman_Klass_Vol"] = 0.5 * (hl**2) - (2 * np.log(2.0) - 1.0) * (co**2)

    out["Vol_Ratio"] = out["Realized_Vol_5"] / out["Realized_Vol_20"]
    return out


# -----------------------------------------------------------------------------
# Group 3 — Technical Indicators (manual)
# -----------------------------------------------------------------------------


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI (Wilder), MACD stack, and Bollinger Band level/squeeze/position
    without TA-Lib — momentum, trend, and mean-reversion context.
    """
    out = df.copy()
    close = out["Close"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha_rsi = 1.0 / 14.0
    avg_gain = gain.ewm(alpha=alpha_rsi, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha_rsi, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
    out.loc[avg_loss == 0.0, "RSI_14"] = 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["BB_Upper"] = ma20 + 2.0 * std20
    out["BB_Lower"] = ma20 - 2.0 * std20
    bb_mid = ma20.replace(0.0, np.nan)
    out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / bb_mid
    band = out["BB_Upper"] - out["BB_Lower"]
    out["BB_Position"] = (close - out["BB_Lower"]) / band.replace(0.0, np.nan)

    return out


# -----------------------------------------------------------------------------
# Group 4 — Momentum and Reversal
# -----------------------------------------------------------------------------


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-horizon price momentum and rate-of-change to capture trend
    persistence and short-term reversal.
    """
    out = df.copy()
    c = out["Close"]
    out["Momentum_5"] = c / c.shift(5) - 1.0
    out["Momentum_20"] = c / c.shift(20) - 1.0
    out["Momentum_60"] = c / c.shift(60) - 1.0
    out["ROC_10"] = (c - c.shift(10)) / c.shift(10)
    return out


# -----------------------------------------------------------------------------
# Group 5 — Volume Features
# -----------------------------------------------------------------------------


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume relative to its recent average (unusual activity) and log volume
    to tame heavy-tailed volume skew.
    """
    out = df.copy()
    vol = out["Volume"]
    out["Volume_MA_Ratio"] = vol / vol.rolling(20).mean()
    out["Volume_Log"] = np.log(vol + 1.0)
    return out


# -----------------------------------------------------------------------------
# Group 6 — HMM Regime Labels
# -----------------------------------------------------------------------------


def _semantic_hmm_mapping(mean_by_state: dict[int, float]) -> dict[int, tuple[int, str]]:
    """
    Map raw HMM states to semantic regime codes and labels from mean log return.

    Highest mean return → bull (1), lowest → bear (2), middle → high_vol (0).
    """
    states = sorted(mean_by_state.keys(), key=lambda s: mean_by_state[s])
    low, mid, high = states[0], states[1], states[2]
    return {
        high: (1, "bull"),
        low: (2, "bear"),
        mid: (0, "high_vol"),
    }


def add_hmm_regimes(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Fit a 3-state Gaussian HMM on [Log_Return, Realized_Vol_20] per ticker,
    assign each day a regime, then remap raw states to bull/bear/high_vol using
    mean log return ordering — captures latent market regimes for conditioning.
    """
    out = df.copy()
    lr = out["Log_Return"]
    rv20 = out["Realized_Vol_20"]
    mask = lr.notna() & rv20.notna()
    X = np.column_stack([lr[mask].to_numpy(dtype=float), rv20[mask].to_numpy(dtype=float)])

    out["HMM_Regime"] = np.nan
    out["HMM_Regime_Label"] = pd.Series(index=out.index, dtype=object)

    if len(X) < 30:
        return out

    lr_vals = lr[mask].to_numpy(dtype=float)
    rv_vals = rv20[mask].to_numpy(dtype=float)
    X_raw = np.column_stack([lr_vals, rv_vals])
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    X_scaled = (X_raw - mu) / sigma

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        random_state=42,
        verbose=False,
    )
    hmm.fit(X_scaled)
    raw_states = hmm.predict(X_scaled)
    idx = out.index[mask]

    if hasattr(hmm, "monitor_") and hmm.monitor_ is not None and not hmm.monitor_.converged:
        print(
            f"  [WARNING] HMM did not converge for {ticker} — regime labels may be unreliable"
        )

    means: dict[int, float] = {}
    for k in (0, 1, 2):
        sel = raw_states == k
        if np.any(sel):
            means[k] = float(np.mean(lr_vals[sel]))
        else:
            means[k] = float("nan")

    raw_to_semantic = _semantic_hmm_mapping(means)
    for row_i, row_idx in enumerate(idx):
        raw_s = int(raw_states[row_i])
        sem_code, label = raw_to_semantic[raw_s]
        out.loc[row_idx, "HMM_Regime"] = sem_code
        out.loc[row_idx, "HMM_Regime_Label"] = label

    # Regime distribution (semantic labels)
    valid = out.loc[idx, "HMM_Regime_Label"].dropna()
    counts = valid.value_counts(normalize=True) * 100.0
    print(f"  [{ticker}] HMM regime distribution (% of days):")
    for name in ("bull", "bear", "high_vol"):
        pct = counts.get(name, 0.0)
        print(f"    {name}: {pct:.2f}%")

    return out


# -----------------------------------------------------------------------------
# Group 7 — Cross-Asset Correlation
# -----------------------------------------------------------------------------


def add_cross_asset_correlation(
    df: pd.DataFrame,
    ticker: str,
    index_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Add 60-day rolling correlation of the stock's log returns with a broad index
    (NIFTY for Indian names, S&P 500 for JPM) — measures equity–macro linkage.
    """
    out = df.copy()
    sym = str(out["Ticker"].iloc[0])
    if INDIAN_TICKER_SUBSTR in sym:
        key = "^NSEI"
    else:
        key = "^GSPC"

    idx_ret = index_returns[key].reindex(pd.to_datetime(out["Date"])).ffill()
    idx_ret.index = out.index
    stock_r = out["Log_Return"]
    out["Rolling_Corr_Index"] = stock_r.rolling(60).corr(idx_ret)
    return out


# -----------------------------------------------------------------------------
# Master pipeline
# -----------------------------------------------------------------------------

ENGINEERED_COLS = (
    [f"Return_Lag_{n}" for n in LAG_DAYS]
    + [
        "Realized_Vol_5",
        "Realized_Vol_20",
        "Realized_Vol_60",
        "Garman_Klass_Vol",
        "Vol_Ratio",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "BB_Position",
        "Momentum_5",
        "Momentum_20",
        "Momentum_60",
        "ROC_10",
        "Volume_MA_Ratio",
        "Volume_Log",
        "HMM_Regime",
        "HMM_Regime_Label",
        "Rolling_Corr_Index",
    ]
)

BASE_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "Log_Return", "Ticker"]


def process_ticker(
    ticker: str,
    df: pd.DataFrame,
    index_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Run all feature groups in order and return the enriched frame (before global
    dropna — caller drops NaNs and saves).
    """
    out = df.copy()
    out = add_lagged_returns(out)
    out = add_volatility_features(out)
    out = add_technical_indicators(out)
    out = add_momentum(out)
    out = add_volume_features(out)
    out = add_hmm_regimes(out, ticker)
    out = add_cross_asset_correlation(out, ticker, index_returns)
    return out


def _output_stem_from_raw_filename(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith("_raw"):
        return stem[: -len("_raw")]
    return stem


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def download_index_returns(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.Series]:
    """
    Download ^NSEI and ^GSPC once, compute daily log returns aligned by calendar date.
    """
    tickers = ["^NSEI", "^GSPC"]
    start_s = start.strftime("%Y-%m-%d")
    end_s = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out: dict[str, pd.Series] = {}
    for t in tickers:
        data = yf.download(t, start=start_s, end=end_s, auto_adjust=True, progress=False)
        data = _flatten_yfinance_columns(data)
        px = data["Close"].dropna()
        lr = np.log(px / px.shift(1))
        lr.name = t
        out[t] = lr
    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for fname in RAW_FILES:
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        d = pd.read_csv(path, parse_dates=["Date"])
        d = d.sort_values("Date").reset_index(drop=True)
        frames.append(d)

    all_dates = pd.concat([f["Date"] for f in frames])
    start = all_dates.min()
    end = all_dates.max()
    print("Downloading index series for cross-asset correlation (NIFTY, S&P 500)...")
    index_returns = download_index_returns(start, end)

    for fname in RAW_FILES:
        stem = _output_stem_from_raw_filename(fname)
        path = RAW_DIR / fname
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        n_orig = len(df)

        enriched = process_ticker(stem, df, index_returns)

        enriched = enriched.dropna(how="any")
        n_after = len(enriched)
        print(
            f"\n[{stem}] rows: {n_orig} original → {n_after} after dropna "
            f"→ {n_orig - n_after} lost"
        )

        feat_count = len(ENGINEERED_COLS)
        print(f"[{stem}] engineered feature columns: {feat_count}")

        enriched = enriched[BASE_COLS + list(ENGINEERED_COLS)]
        out_path = PROCESSED_DIR / f"{stem}_features.csv"
        enriched.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
