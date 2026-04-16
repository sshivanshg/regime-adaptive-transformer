"""
RAMT interactive simulation dashboard — modular Streamlit app with date scrubbing,
shadow portfolio conviction, regime-shaded equity, and tabbed analytics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.ramt.dataset import (
    ALL_FEATURE_COLS,
    MACRO_COLS,
    PRICE_COLS,
    TECH_COLS,
    TICKER_TO_ID,
    VOLUME_COLS,
)
from models.ramt.model import build_ramt

try:
    from models.ramt.dataset import build_ticker_universe
except Exception:  # pragma: no cover
    build_ticker_universe = None  # type: ignore[misc, assignment]

# ─── Calendar bounds (user requirement) ─────────────────────
WINDOW_MIN = pd.Timestamp("2023-01-01")
WINDOW_MAX = pd.Timestamp("2026-12-31")

REGIME_COLORS = {0: "#eab308", 1: "#22c55e", 2: "#ef4444"}
REGIME_NAMES = {0: "HIGH VOL", 1: "BULL", 2: "BEAR"}

# Light tints for chart backgrounds (High-Vol = yellow, Bull = green, Bear = red)
REGIME_VRECT_FILL = {
    0: "rgba(250, 204, 21, 0.22)",   # light yellow — high vol
    1: "rgba(34, 197, 94, 0.14)",   # light green — bull
    2: "rgba(248, 113, 113, 0.18)", # light red — bear
}

TICKERS_FALLBACK = {
    "TCS": "TCS_NS",
    "RELIANCE": "RELIANCE_NS",
    "HDFC Bank": "HDFCBANK_NS",
    "EPIGRAL": "EPIGRAL_NS",
    "JPM (US)": "JPM",
}


# ─── Page & theme ────────────────────────────────────────────
def _inject_theme_css() -> None:
    mono = "'JetBrains Mono', 'SF Mono', ui-monospace, monospace"
    st.markdown(
        f"""
<style>
    html, body, [class*="css"] {{
        font-family: {mono};
    }}
    .stApp, .main, header[data-testid="stHeader"] {{
        background-color: #0b1020 !important;
    }}
    .block-container {{
        padding-top: 1.2rem;
        max-width: 1400px;
    }}
    div[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #0b1020 100%);
        border-right: 1px solid #1e293b;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 26px;
        font-weight: 600;
        color: #e2e8f0;
        font-family: {mono};
    }}
    div[data-testid="stMetricLabel"] {{
        color: #94a3b8;
        font-size: 11px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .shadow-badge-bear {{
        display: inline-block;
        background: #450a0a;
        color: #fecaca;
        border: 1px solid #7f1d1d;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
    }}
    h1, h2, h3 {{
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: #38bdf8 !important;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="RAMT · Production Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_theme_css()


# ─── Data layer ──────────────────────────────────────────────
def _ranking_csv_mtime(path: Path) -> float:
    return float(path.stat().st_mtime) if path.is_file() else 0.0


@st.cache_data(show_spinner=False)
def _read_ranking_predictions_cached(csv_path: str, mtime: float) -> pd.DataFrame | None:
    """Single read of ranking CSV; cache invalidates when mtime changes."""
    p = Path(csv_path)
    if not p.is_file():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def _ensure_nifty_features_cached(processed_s: str, raw_s: str) -> str | None:
    """Build or resolve NIFTY features Parquet (HMM regime)."""
    try:
        from models.backtest import ensure_nifty_features_parquet

        return ensure_nifty_features_parquet(processed_s, raw_s)
    except Exception:
        out = Path(processed_s) / "_NSEI_features.parquet"
        return str(out) if out.is_file() else None


@st.cache_data(show_spinner=False)
def _load_nifty_regime_series_cached(features_parquet: str | None) -> pd.Series | None:
    if not features_parquet:
        return None
    p = Path(features_parquet)
    if not p.is_file():
        return None
    df = pd.read_parquet(p, columns=["Date", "HMM_Regime"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["HMM_Regime"].astype(float).sort_index().ffill()


@st.cache_data(show_spinner=False)
def _load_nifty_adj_close_cached(raw_dir_s: str) -> pd.Series | None:
    try:
        from models.backtest import _load_nifty_benchmark_raw
    except Exception:
        return None
    try:
        raw = _load_nifty_benchmark_raw(Path(raw_dir_s))
    except Exception:
        return None
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw.set_index("Date")["Adj Close"].astype(float).sort_index()


@st.cache_data(show_spinner=False)
def discover_ticker_stems() -> list[str]:
    if build_ticker_universe is not None:
        try:
            u = build_ticker_universe(str(ROOT / "data" / "processed"))
            if u:
                return sorted(u)
        except Exception:
            pass
    processed = ROOT / "data" / "processed"
    if not processed.is_dir():
        return sorted(TICKERS_FALLBACK.values())
    out: list[str] = []
    for p in sorted(processed.glob("*_features.parquet")):
        stem = p.stem[: -len("_features")] if p.stem.endswith("_features") else p.stem
        if stem.lstrip("_").upper() in ("NSEI", "NIFTY50"):
            continue
        if stem.startswith("macro_"):
            continue
        out.append(stem)
    return sorted(set(out)) if out else sorted(TICKERS_FALLBACK.values())


@st.cache_data(show_spinner=False)
def load_features(ticker_code: str) -> pd.DataFrame | None:
    pq = ROOT / f"data/processed/{ticker_code}_features.parquet"
    csv = ROOT / f"data/processed/{ticker_code}_features.csv"
    path = pq if pq.exists() else csv
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df = pd.read_csv(path, parse_dates=["Date"])
    if "Date" in df.columns:
        df = df.sort_values("Date").set_index("Date")
    return df


@st.cache_data(show_spinner=False)
def load_predictions(model_name: str) -> pd.DataFrame | None:
    m = (model_name or "").lower()
    if m == "ramt":
        path = ROOT / "results/ramt_predictions.csv"
        if not path.exists():
            path = ROOT / "results/ranking_predictions.csv"
    else:
        path = ROOT / f"results/{m}_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    if "predicted_alpha" in df.columns and "actual_alpha" in df.columns:
        df = df.rename(columns={"predicted_alpha": "y_pred", "actual_alpha": "y_true"})
    return df


@st.cache_data(show_spinner=False)
def load_all_predictions() -> dict[str, pd.DataFrame]:
    models: dict[str, pd.DataFrame] = {}
    for name in ["xgboost", "lstm", "ramt"]:
        df = load_predictions(name)
        if df is not None and {"y_true", "y_pred"}.issubset(df.columns):
            models[name.upper()] = df
    return models


@st.cache_data(show_spinner=False)
def load_monthly_rankings() -> pd.DataFrame | None:
    path = ROOT / "results/monthly_rankings.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_backtest_results() -> pd.DataFrame | None:
    path = ROOT / "results/backtest_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date")


@st.cache_data(show_spinner=False)
def load_data(root_s: str) -> dict[str, Any]:
    """
    Central loader: ensures NIFTY features exist, exposes paths and raw frames for the session.
    Cached so scrubbing the UI does not re-hit disk beyond inner cached readers.
    """
    root = Path(root_s)
    processed = root / "data" / "processed"
    raw = root / "data" / "raw"
    nifty_parquet = _ensure_nifty_features_cached(str(processed), str(raw))
    regime_series = _load_nifty_regime_series_cached(nifty_parquet)
    nifty_prices = _load_nifty_adj_close_cached(str(raw))

    rpath = root / "results" / "ranking_predictions.csv"
    mtime = _ranking_csv_mtime(rpath)
    ranking_df = _read_ranking_predictions_cached(str(rpath), mtime)

    return {
        "root": root,
        "nifty_features_path": nifty_parquet,
        "regime_series": regime_series,
        "nifty_prices": nifty_prices,
        "ranking_predictions_path": rpath,
        "ranking_mtime": mtime,
        "ranking_df": ranking_df,
        "monthly_rankings": load_monthly_rankings(),
        "backtest_results": load_backtest_results(),
    }


# ─── Filters & simulation ────────────────────────────────────
def filter_ranking_by_dates(
    df: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    test_only: bool,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    end_inc = end.normalize() + pd.Timedelta(days=1)
    out = out[(out["Date"] >= start.normalize()) & (out["Date"] < end_inc)]
    if test_only and "Period" in out.columns:
        out = out[out["Period"].astype(str).str.strip() == "Test"]
    return out


def _regime_allocation_gear(reg: int | None, top_n: int) -> tuple[float, int]:
    """BULL 100% / HIGH_VOL 50%×3 / BEAR 20%×5; ``reg is None`` → legacy full top_n."""
    if reg is None:
        return 1.0, top_n
    if reg == 0:
        return 0.5, min(3, top_n)
    if reg == 2:
        return 0.2, min(5, top_n)
    return 1.0, top_n


def simulate_portfolio(
    full: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int,
    initial_capital: float,
    test_only: bool,
    nifty_prices: pd.Series | None,
    regime_series: pd.Series | None = None,
) -> dict[str, Any] | None:
    need = {"Date", "Ticker", "predicted_alpha", "actual_alpha"}
    if not need.issubset(full.columns):
        return None
    df = filter_ranking_by_dates(full, start, end, test_only)
    if df.empty:
        return {
            "rebalance_dates": [],
            "strategy_equity": np.array([]),
            "nifty_equity": np.array([]),
            "period_returns": np.array([]),
            "rows_detail": pd.DataFrame(),
        }

    reb_dates = sorted(df["Date"].unique())
    period_excess: list[float] = []
    rows_detail: list[dict[str, Any]] = []
    for d in reb_dates:
        ts = pd.Timestamp(d)
        reg = regime_at(regime_series, ts) if regime_series is not None else None
        alloc, n_pick = _regime_allocation_gear(reg, top_n)
        block = df[df["Date"] == d].nlargest(n_pick, "predicted_alpha")
        if block.empty:
            continue
        m_excess = float(block["actual_alpha"].mean())
        period_ret = float(m_excess * alloc)
        if reg == 2:
            period_ret = float(max(period_ret, -0.05))
        period_excess.append(period_ret)
        tickers = block["Ticker"].astype(str).tolist()
        rows_detail.append(
            {
                "Date": d,
                "top_tickers": ", ".join(tickers),
                "tickers_list": tickers,
                "mean_actual_alpha": m_excess,
                "n_names": len(block),
                "regime": reg,
                "allocation": alloc,
                "period_return": period_ret,
            }
        )

    if not period_excess:
        return {
            "rebalance_dates": [],
            "strategy_equity": np.array([]),
            "nifty_equity": np.array([]),
            "period_returns": np.array([]),
            "rows_detail": pd.DataFrame(),
        }

    period_excess = np.asarray(period_excess, dtype=np.float64)
    strat_eq = float(initial_capital) * np.cumprod(1.0 + period_excess)

    nifty_eq = np.full(len(reb_dates), np.nan, dtype=np.float64)
    if nifty_prices is not None and len(reb_dates) > 0:
        p0 = float(nifty_prices.asof(reb_dates[0]))
        if p0 > 0 and np.isfinite(p0):
            for i, d in enumerate(reb_dates):
                pi = float(nifty_prices.asof(pd.Timestamp(d)))
                nifty_eq[i] = float(initial_capital) * (pi / p0)

    return {
        "rebalance_dates": reb_dates,
        "strategy_equity": strat_eq,
        "nifty_equity": nifty_eq,
        "period_returns": period_excess,
        "rows_detail": pd.DataFrame(rows_detail),
    }


def regime_at(reg: pd.Series | None, ts: pd.Timestamp) -> int | None:
    if reg is None:
        return None
    v = reg.asof(ts)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return int(max(0, min(2, int(float(v)))))


def top5_shadow_for_date(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Top 5 by predicted_alpha on the given rebalance date (nearest prior date if missing)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    ts = pd.Timestamp(as_of).normalize()
    exact = d[d["Date"].dt.normalize() == ts]
    if exact.empty:
        prior = d[d["Date"].dt.normalize() <= ts]
        if prior.empty:
            return pd.DataFrame()
        last_d = prior["Date"].max()
        exact = d[d["Date"] == last_d]
    return exact.nlargest(5, "predicted_alpha")[
        ["Ticker", "predicted_alpha", "actual_alpha"]
    ].reset_index(drop=True)


# ─── Metrics ─────────────────────────────────────────────────
def directional_accuracy_rows(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 1:
        return 0.0
    a = df["actual_alpha"].to_numpy(dtype=float)
    p = df["predicted_alpha"].to_numpy(dtype=float)
    return float(np.mean(np.sign(a) == np.sign(p)) * 100.0)


def sharpe_from_period_returns(period: np.ndarray) -> float:
    if period is None or len(period) < 2:
        return 0.0
    m = float(np.mean(period))
    s = float(np.std(period, ddof=0))
    if s < 1e-12:
        return 0.0
    # Monthly rebalance cadence → √12 annualization
    return m / s * np.sqrt(12.0)


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    if equity is None or len(equity) < 1:
        return 0.0
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())


def calculate_metrics(
    ranking_slice: pd.DataFrame,
    sim: dict[str, Any] | None,
) -> dict[str, float]:
    """Directional accuracy from all ranking rows; Sharpe & MDD from simulated equity."""
    da = directional_accuracy_rows(ranking_slice)
    pr = sim.get("period_returns") if sim else None
    eq = sim.get("strategy_equity") if sim else None
    if pr is None or len(pr) == 0:
        sharpe = 0.0
        mdd = 0.0
    else:
        sharpe = sharpe_from_period_returns(np.asarray(pr))
        mdd = max_drawdown_from_equity(np.asarray(eq)) if eq is not None and len(eq) else 0.0
    return {"DA%": da, "Sharpe": sharpe, "MaxDD": mdd}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Legacy daily-style metrics for model comparison tables."""
    if len(y_true) == 0:
        return {}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    da = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)
    rolling_std = pd.Series(y_pred).rolling(20).std()
    rolling_std = rolling_std.fillna(float(np.std(y_pred)))
    position = np.clip(y_pred / (rolling_std.values + 1e-8), -2, 2)
    strategy_ret = y_true * position
    sharpe = float(np.mean(strategy_ret) / (np.std(strategy_ret) + 1e-8) * np.sqrt(252))
    cumulative = np.cumprod(1 + strategy_ret)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / (rolling_max + 1e-8)
    max_dd = float(drawdown.min())
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "DA%": round(da, 2),
        "Sharpe": round(sharpe, 2),
        "MaxDD": round(max_dd, 4),
    }


# ─── Sidebar ─────────────────────────────────────────────────
@dataclass
class SidebarState:
    ticker_code: str
    selected_model: str
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    as_of_date: pd.Timestamp
    live_test_only: bool
    sim_top_n: int
    sim_capital: float
    chart_days: int


def _clamp_window(
    dmin: pd.Timestamp | None,
    dmax: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Intersection of dataset dates with [2023-01-01, 2026-12-31]."""
    lo, hi = WINDOW_MIN, WINDOW_MAX
    if dmin is not None:
        lo = max(lo, pd.Timestamp(dmin))
    if dmax is not None:
        hi = min(hi, pd.Timestamp(dmax))
    if lo > hi:
        # No overlap between file dates and 2023–2026 — keep calendar bounds for UI
        return WINDOW_MIN, WINDOW_MAX
    return lo, hi


def render_sidebar(
    ranking_df: pd.DataFrame | None,
) -> SidebarState:
    stems = discover_ticker_stems()
    dmin = dmax = None
    if ranking_df is not None and not ranking_df.empty:
        ranking_df = ranking_df.copy()
        ranking_df["Date"] = pd.to_datetime(ranking_df["Date"])
        dmin = ranking_df["Date"].min()
        dmax = ranking_df["Date"].max()

    w_lo, w_hi = _clamp_window(dmin, dmax)

    with st.sidebar:
        st.markdown("## ◈ RAMT Workbench")
        st.caption("Regime-adaptive simulation · NIFTY200")
        st.markdown("---")

        ticker_code = st.selectbox(
            "Ticker (processed stem)",
            stems,
            index=0,
            help="Matches `data/processed/{stem}_features.parquet`.",
        )

        available_models: list[str] = []
        for name in ["xgboost", "lstm", "ramt"]:
            df = load_predictions(name)
            if df is not None and {"y_true", "y_pred"}.issubset(df.columns):
                available_models.append(name.upper())

        if available_models:
            selected_model = st.selectbox("Primary model", available_models, index=0)
        else:
            selected_model = "XGBOOST"
            st.warning("No `results/*_predictions.csv` found.")

        st.markdown("---")
        st.markdown("### Date window (2023–2026 blind test)")

        if dmin is None:
            st.caption("Add `results/ranking_predictions.csv` to enable scrubbing.")
            ds, de = w_lo.date(), w_hi.date()
            live_test_only = True
            date_start = pd.Timestamp(ds)
            date_end = pd.Timestamp(de)
        else:
            live_test_only = st.checkbox(
                "Blind test rows only (`Period == Test`)",
                value=True,
                help="Uncheck to include train-era rows in the window.",
            )
            all_reb = sorted(
                {pd.Timestamp(d).normalize() for d in pd.to_datetime(ranking_df["Date"])}
            )
            all_reb = [d for d in all_reb if w_lo <= d <= w_hi]
            if len(all_reb) >= 2:
                n1 = len(all_reb) - 1
                idx_pair = st.slider(
                    "Scrub rebalance window (cached metrics & equity)",
                    min_value=0,
                    max_value=n1,
                    value=(0, n1),
                    help="Slide both ends — updates DA, Sharpe, max DD, and equity instantly.",
                )
                i0, i1 = int(idx_pair[0]), int(idx_pair[1])
                if i0 > i1:
                    i0, i1 = i1, i0
                date_start = all_reb[i0]
                date_end = all_reb[i1]
            else:
                yr_a = max(2023, w_lo.year)
                yr_b = min(2026, w_hi.year)
                if yr_a > yr_b:
                    yr_a, yr_b = yr_b, yr_a
                yr_rng = st.slider(
                    "Year range",
                    min_value=2023,
                    max_value=2026,
                    value=(yr_a, yr_b),
                )
                date_start = pd.Timestamp(f"{yr_rng[0]}-01-01")
                date_end = pd.Timestamp(f"{yr_rng[1]}-12-31")
                date_start = max(date_start, w_lo)
                date_end = min(date_end, w_hi)
                if date_start > date_end:
                    date_start, date_end = date_end, date_start
                st.caption("Fine-tune with calendar (optional)")
                cal = st.date_input(
                    "Calendar range",
                    value=(date_start.date(), date_end.date()),
                    min_value=w_lo.date(),
                    max_value=w_hi.date(),
                )
                if isinstance(cal, tuple):
                    date_start = pd.Timestamp(cal[0])
                    date_end = pd.Timestamp(cal[1])
                else:
                    date_start = date_end = pd.Timestamp(cal)
                if date_start > date_end:
                    date_start, date_end = date_end, date_start

        # As-of date for shadow picks: rebalance dates inside window
        if ranking_df is not None and not ranking_df.empty:
            udates = sorted(
                {pd.Timestamp(d).normalize() for d in pd.to_datetime(ranking_df["Date"])}
            )
            udates = [d for d in udates if date_start <= d <= date_end]
            if not udates:
                udates = sorted(
                    {pd.Timestamp(d).normalize() for d in pd.to_datetime(ranking_df["Date"])}
                )
            default_asof = udates[-1] if udates else date_end
            as_of = st.select_slider(
                "As-of date (shadow picks)",
                options=udates,
                value=default_asof,
                format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"),
                help="Top-5 conviction uses this rebalance date.",
            )
            as_of_date = pd.Timestamp(as_of)
        else:
            as_of_date = date_end
            st.caption("As-of defaults to window end when predictions load.")

        st.markdown("---")
        st.markdown("### Simulation")
        sim_top_n = st.slider("Top-N per rebalance", 3, 10, 5)
        sim_capital = float(
            st.number_input("Starting capital (₹)", value=100_000.0, step=10_000.0)
        )

        st.markdown("---")
        chart_days = st.slider("Chart history (days)", 30, 365, 120, step=30)

        st.markdown("---")
        if st.button("Refresh cache", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    return SidebarState(
        ticker_code=ticker_code,
        selected_model=selected_model,
        date_start=date_start,
        date_end=date_end,
        as_of_date=as_of_date,
        live_test_only=live_test_only,
        sim_top_n=sim_top_n,
        sim_capital=sim_capital,
        chart_days=chart_days,
    )


# ─── Charts ──────────────────────────────────────────────────
def render_charts(
    sim: dict[str, Any] | None,
    regime_series: pd.Series | None,
) -> None:
    fig = build_live_equity_figure(sim, regime_series)
    if fig is None:
        st.info("No rebalances in this window — widen the date range or disable test-only.")
        return
    st.plotly_chart(fig, width="stretch")


def build_live_equity_figure(
    sim: dict[str, Any] | None,
    regime_series: pd.Series | None,
) -> go.Figure | None:
    if not sim:
        return None
    dates = sim.get("rebalance_dates") or []
    strat = sim.get("strategy_equity")
    if not dates or strat is None or len(strat) == 0:
        return None
    nifty = sim.get("nifty_equity")
    detail = sim.get("rows_detail")
    if detail is not None and not detail.empty and "tickers_list" in detail.columns:
        hover_holdings = [" · ".join(t) for t in detail["tickers_list"]]
    elif detail is not None and not detail.empty and "top_tickers" in detail.columns:
        hover_holdings = detail["top_tickers"].tolist()
    else:
        hover_holdings = [""] * len(dates)

    fig = go.Figure()

    if regime_series is not None and len(dates) >= 2:
        for i in range(len(dates) - 1):
            ts = pd.Timestamp(dates[i])
            ri = regime_at(regime_series, ts)
            if ri is None:
                ri = 1
            fig.add_vrect(
                x0=dates[i],
                x1=dates[i + 1],
                fillcolor=REGIME_VRECT_FILL.get(ri, REGIME_VRECT_FILL[1]),
                line_width=0,
                layer="below",
            )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strat,
            mode="lines+markers",
            name="RAMT top-N (compounded α)",
            line=dict(color="#4ade80", width=2.5),
            marker=dict(size=7),
            customdata=np.array([[h] for h in hover_holdings], dtype=object),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Equity: %{y:,.0f} ₹<br>"
                "<b>Holdings (top-N)</b><br>%{customdata[0]}<extra></extra>"
            ),
        )
    )
    if nifty is not None and len(nifty) == len(dates) and np.any(np.isfinite(nifty)):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=nifty,
                mode="lines+markers",
                name="NIFTY buy & hold",
                line=dict(color="#38bdf8", width=2, dash="dash"),
                marker=dict(size=5),
                hovertemplate="%{x|%Y-%m-%d}<br>NIFTY notional: %{y:,.0f} ₹<extra></extra>",
            )
        )

    fig.update_layout(
        title="Equity curve — HMM regime shading (yellow=high vol, green=bull, red=bear)",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="'JetBrains Mono', monospace", size=12),
        height=480,
        margin=dict(t=48, b=52, l=12, r=12),
        yaxis_title="Notional (₹)",
        xaxis_title="Rebalance date",
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1e293b", rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def plot_actual_vs_predicted(predictions_df: pd.DataFrame | None, ticker_code: str, days: int = 120):
    if predictions_df is None:
        return None
    df = predictions_df[predictions_df["Ticker"] == ticker_code].copy().sort_values("Date").tail(days)
    if df.empty:
        return None
    
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Daily returns — actual vs predicted ({days} d)",
            "Direction match",
        ),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["y_true"] * 100,
            name="Actual %",
            marker_color=["#22c55e" if v > 0 else "#ef4444" for v in df["y_true"]],
            opacity=0.65,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["y_pred"] * 100,
            name="Predicted %",
            line=dict(color="#f97316", width=2),
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", row=1, col=1)
    correct = (np.sign(df["y_true"].values) == np.sign(df["y_pred"].values)).astype(int)
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=correct,
            marker_color=["#22c55e" if c == 1 else "#ef4444" for c in correct],
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0", size=11),
        height=500,
        margin=dict(t=40, b=20),
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def plot_cumulative_returns(predictions_df: pd.DataFrame | None, ticker_code: str):
    if predictions_df is None:
        return None
    df = predictions_df[predictions_df["Ticker"] == ticker_code].copy().sort_values("Date")
    if df.empty:
        return None
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    bah = np.cumprod(1 + y_true) - 1
    position = np.sign(y_pred)
    position[position == 0] = 0
    strategy = np.cumprod(1 + y_true * position) - 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
        y=strategy * 100,
            name="Strategy",
            line=dict(color="#4ade80", width=2),
            fill="tozeroy",
            fillcolor="rgba(74, 222, 128, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
        y=bah * 100,
            name="Buy & hold",
            line=dict(color="#38bdf8", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        height=340,
        yaxis_title="Cumulative %",
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def plot_regime_history(features_df: pd.DataFrame | None, days: int = 180):
    if features_df is None or "HMM_Regime" not in features_df.columns:
        return None
    df = features_df.tail(days).copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Price · regime tint", "Regime"),
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )
    if "Close" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Close",
                line=dict(color="#e2e8f0", width=1.2),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=[1] * len(df),
            marker_color=[REGIME_COLORS[int(r)] for r in df["HMM_Regime"]],
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        height=460,
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def _ranking_metrics(act: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    if len(act) < 2:
        return {}
    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    mae = float(np.mean(np.abs(act - pred)))
    da = float(np.mean(np.sign(act) == np.sign(pred)) * 100)
    corr = float(np.corrcoef(act, pred)[0, 1])
    return {"RMSE": round(rmse, 5), "MAE": round(mae, 5), "DA%": round(da, 2), "ρ": round(corr, 3), "n": int(len(act))}


def plot_ramt_ranking_predicted_vs_actual(
    rank_df: pd.DataFrame, ticker_code: str
) -> tuple[go.Figure | None, dict[str, float | int]]:
    if rank_df is None or rank_df.empty:
        return None, {}
    need = {"Date", "predicted_alpha", "actual_alpha", "Ticker"}
    if not need.issubset(rank_df.columns):
        return None, {}
    d = rank_df[rank_df["Ticker"] == ticker_code].copy().sort_values("Date")
    if d.empty:
        return None, {}
    act = d["actual_alpha"].to_numpy(dtype=float)
    pred = d["predicted_alpha"].to_numpy(dtype=float)
    metrics = _ranking_metrics(act, pred)
    act_pct = act * 100.0
    pred_pct = pred * 100.0
    dates = d["Date"]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.54, 0.46],
        vertical_spacing=0.18,
        subplot_titles=(
            f"{ticker_code} · realized vs predicted α",
            "Calibration",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=act_pct,
            name="Realized α",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pred_pct,
            name="Predicted α",
            mode="lines+markers",
            line=dict(color="#14b8a6", width=2),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )
    lo = float(min(act_pct.min(), pred_pct.min()))
    hi = float(max(act_pct.max(), pred_pct.max()))
    pad = max((hi - lo) * 0.12, 0.25)
    lim_lo, lim_hi = lo - pad, hi + pad
    fig.add_trace(
        go.Scatter(
            x=[lim_lo, lim_hi],
            y=[lim_lo, lim_hi],
            mode="lines",
            name="y = x",
            line=dict(color="#64748b", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=act_pct,
            y=pred_pct,
            mode="markers",
            marker=dict(size=8, color="#a78bfa", line=dict(width=1, color="#e9d5ff")),
            text=dates.dt.strftime("%Y-%m-%d"),
            hovertemplate="Realized %{x:.3f}% · Pred %{y:.3f}% · %{text}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=720,
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=56, r=32, t=80, b=96),
    )
    fig.update_xaxes(gridcolor="#1e293b", row=1, col=1)
    fig.update_yaxes(gridcolor="#1e293b", row=1, col=1)
    fig.update_xaxes(title="Realized α %", gridcolor="#1e293b", row=2, col=1, range=[lim_lo, lim_hi])
    fig.update_yaxes(title="Predicted α %", gridcolor="#1e293b", row=2, col=1, range=[lim_lo, lim_hi])
    return fig, metrics


def plot_metrics_comparison(all_predictions: dict[str, pd.DataFrame], ticker_code: str):
    if not all_predictions:
        return None
    rows: list[dict[str, Any]] = []
    for model_name, pred_df in all_predictions.items():
        t_df = pred_df[pred_df["Ticker"] == ticker_code]
        if t_df.empty:
            continue
        m = compute_metrics(t_df["y_true"].values, t_df["y_pred"].values)
        m["Model"] = model_name
        rows.append(m)
    if not rows:
        return None
    metrics_df = pd.DataFrame(rows)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Directional accuracy %", "Sharpe"),
    )
    colors = ["#38bdf8", "#4ade80", "#f97316"]
    fig.add_trace(
        go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["DA%"],
            marker_color=colors[: len(metrics_df)],
            text=metrics_df["DA%"].round(2),
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#f87171", row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["Sharpe"],
            marker_color=colors[: len(metrics_df)],
            text=metrics_df["Sharpe"].round(2),
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        height=300,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def get_current_regime(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or "HMM_Regime" not in df.columns:
        return None
    last_regime = int(df["HMM_Regime"].iloc[-1])
    regimes = df["HMM_Regime"].values
    days_in_regime = 1
    for i in range(len(regimes) - 2, -1, -1):
        if int(regimes[i]) == last_regime:
            days_in_regime += 1
        else:
            break
    recent = df["HMM_Regime"].tail(60)
    total = len(recent)
    bull_pct = float((recent == 1).sum() / total * 100)
    bear_pct = float((recent == 2).sum() / total * 100)
    hv_pct = float((recent == 0).sum() / total * 100)
    return {
        "current": last_regime,
        "name": REGIME_NAMES[last_regime],
        "color": REGIME_COLORS[last_regime],
        "days": days_in_regime,
        "bull_pct": bull_pct,
        "bear_pct": bear_pct,
        "hv_pct": hv_pct,
    }


# ─── Live market pulse & RAMT inference ────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_macro_data_cached(ticker_stem: str, root_s: str) -> dict[str, Any]:
    """Hourly cache — keeps demos snappy on repeated clicks."""
    from dashboard.market_scraper import fetch_live_macro_data_engine

    return fetch_live_macro_data_engine(ticker_stem, Path(root_s))


def _load_stock_raw_history(stem: str, root: Path) -> pd.DataFrame | None:
    pq = root / "data" / "raw" / f"{stem}.parquet"
    csv = root / "data" / "raw" / f"{stem}_raw.csv"
    path = pq if pq.is_file() else csv
    if not path.is_file():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["Date"])
    if df.empty or "Date" not in df.columns:
        return None
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"].astype(float)
    if "Adj Close" not in df.columns or "Volume" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_live_stock_feature_history(stem: str, root: Path) -> pd.DataFrame | None:
    """
    Recompute stock-only price/technical/volume features from the freshest raw bars.

    Processed feature files stop before the latest dates because the supervised targets drop
    the unlabeled tail, so the live predictor needs a raw-price bridge.
    """
    raw = _load_stock_raw_history(stem, root)
    if raw is None or raw.empty:
        return None

    px = raw["Adj Close"].astype(float).replace(0.0, np.nan)
    vol = raw["Volume"].astype(float)
    out = raw[["Date"]].copy()

    r1 = px / px.shift(1)
    r5 = px / px.shift(5)
    r21 = px / px.shift(21)
    out["Ret_1d"] = np.log(r1.where(r1 > 0.0))
    out["Ret_5d"] = np.log(r5.where(r5 > 0.0))
    out["Ret_21d"] = np.log(r21.where(r21 > 0.0))

    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / 14.0
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["RSI_14"] = (100.0 - (100.0 / (1.0 + rs))).where(avg_loss != 0.0, 100.0)

    ma20 = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    denom = (2.0 * std20).replace(0.0, np.nan)
    out["BB_Dist"] = (px - ma20) / denom

    sma20 = vol.rolling(20).mean().replace(0.0, np.nan)
    out["Volume_Surge"] = vol / sma20
    return out


def compute_live_stock_feature_snapshot(stem: str, root: Path) -> dict[str, Any] | None:
    need = list(PRICE_COLS + TECH_COLS + VOLUME_COLS)

    raw_hist = compute_live_stock_feature_history(stem, root)
    if raw_hist is not None and not raw_hist.empty:
        ready = raw_hist.dropna(subset=need, how="any")
        if not ready.empty:
            last = ready.iloc[-1]
            return {
                "features": last[need],
                "as_of": pd.Timestamp(last["Date"]),
                "source": "raw",
            }

    pq = root / "data" / "processed" / f"{stem}_features.parquet"
    if not pq.is_file():
        return None
    df = pd.read_parquet(pq)
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    last = df.iloc[-1]
    if not all(c in last.index for c in need):
        return None
    return {
        "features": last[need],
        "as_of": pd.Timestamp(last["Date"]),
        "source": "processed",
    }


def compute_live_stock_feature_row(stem: str, root: Path) -> pd.Series | None:
    snap = compute_live_stock_feature_snapshot(stem, root)
    if snap is None:
        return None
    return snap["features"]


def latest_stock_regime(stem: str, root: Path) -> int | None:
    pq = root / "data" / "processed" / f"{stem}_features.parquet"
    if not pq.is_file():
        return None
    df = pd.read_parquet(pq, columns=["HMM_Regime"])
    if df.empty:
        return None
    return int(np.clip(int(round(float(df["HMM_Regime"].iloc[-1]))), 0, 2))


def _log_return_to_pct(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float((np.exp(float(value)) - 1.0) * 100.0)


def build_live_ramt_sequence(
    ticker_stem: str,
    root: Path,
    macro_pack: dict[str, Any],
) -> tuple[np.ndarray, int] | None:
    """
    Build raw (unscaled) 30×F feature tensor: 29 historical rows from Parquet + 1 live row.
    Live row = latest stock features (Parquet last-known) + macro from macro_pack (training-aligned).
    """
    pq = root / "data" / "processed" / f"{ticker_stem}_features.parquet"
    if not pq.is_file():
        return None
    raw = pd.read_parquet(pq)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date")
    if len(raw) < 30:
        return None
    miss = [c for c in ALL_FEATURE_COLS if c not in raw.columns]
    if miss:
        return None

    seq_df = raw[["Date"] + ALL_FEATURE_COLS].copy()
    stock_hist = compute_live_stock_feature_history(ticker_stem, root)
    if stock_hist is not None and not stock_hist.empty:
        seq_df = seq_df.set_index("Date")
        stock_hist = stock_hist.set_index("Date")
        union_idx = seq_df.index.union(stock_hist.index).sort_values()
        seq_df = seq_df.reindex(union_idx)
        for c in PRICE_COLS + TECH_COLS + VOLUME_COLS:
            seq_df.loc[stock_hist.index, c] = stock_hist[c]
        seq_df.loc[:, MACRO_COLS] = seq_df.loc[:, MACRO_COLS].ffill()
        seq_df = seq_df.reset_index().rename(columns={"index": "Date"})
    if len(seq_df) < 30:
        return None

    base = seq_df[ALL_FEATURE_COLS].tail(30).to_numpy(dtype=np.float32)
    feats = macro_pack.get("macro_features") or {}
    srow = compute_live_stock_feature_row(ticker_stem, root)

    idx = {c: i for i, c in enumerate(ALL_FEATURE_COLS)}
    live = base[-1].copy()

    if srow is not None:
        for c in PRICE_COLS + TECH_COLS + VOLUME_COLS:
            live[idx[c]] = float(srow[c])

    for c in MACRO_COLS:
        v = feats.get(c, float("nan"))
        if not np.isfinite(v):
            v = float(base[-1, idx[c]])
        live[idx[c]] = v

    seq = np.vstack([base[-30:-1], live.reshape(1, -1)])
    regime = int(np.clip(int(round(float(raw["HMM_Regime"].iloc[-1]))), 0, 2))
    return seq.astype(np.float32), regime


def apply_predictor_form_to_last_row(
    seq_unscaled: np.ndarray,
    ret_1d_pct: float,
    mom_20_pct: float,
    rsi: float,
    bb_dist: float,
    vol_surge: float,
) -> np.ndarray:
    """Optional manual tweaks — map % returns to log features like the pipeline."""
    out = seq_unscaled.copy()
    last = out[-1].copy()
    idx = {c: i for i, c in enumerate(ALL_FEATURE_COLS)}
    r1 = ret_1d_pct / 100.0
    last[idx["Ret_1d"]] = float(np.log1p(r1)) if r1 > -0.999 else last[idx["Ret_1d"]]
    m20 = mom_20_pct / 100.0
    last[idx["Ret_21d"]] = float(np.log1p(m20)) if m20 > -0.999 else last[idx["Ret_21d"]]
    last[idx["RSI_14"]] = float(rsi)
    last[idx["BB_Dist"]] = float(bb_dist)
    last[idx["Volume_Surge"]] = float(vol_surge)
    out[-1] = last
    return out


def impute_missing_with_training_medians(seq_unscaled: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Fill NaN / non-finite values using training-set medians from RobustScaler.center_
    (or StandardScaler.mean_ if present). Keeps X aligned with what the scaler expects.
    """
    out = np.asarray(seq_unscaled, dtype=np.float64).copy()
    _, n_cols = out.shape
    fill = np.zeros(n_cols, dtype=np.float64)
    center = getattr(scaler, "center_", None)
    if center is not None:
        fill = np.asarray(center, dtype=np.float64).ravel()
    elif getattr(scaler, "mean_", None) is not None:
        fill = np.asarray(scaler.mean_, dtype=np.float64).ravel()
    if fill.size == 0:
        fill = np.zeros(n_cols, dtype=np.float64)
    elif fill.size < n_cols:
        fill = np.resize(fill, n_cols)
    else:
        fill = fill[:n_cols]
    for j in range(n_cols):
        col = out[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            out[bad, j] = fill[j]
    return out.astype(np.float32)


def run_ramt_live_predict(
    ticker_stem: str,
    root: Path,
    macro_pack: dict[str, Any],
    ret_1d_pct: float,
    mom_20_pct: float,
    rsi: float,
    bb_dist: float,
    vol_surge: float,
    regime_id: int,
) -> tuple[float | None, float | None, str]:
    """
    Load `results/ramt_scaler.joblib`, impute missing values with training medians,
    transform the 10 RAMT feature columns, then `model.forward` (regime is separate
    from X — not scaled). Returns (monthly_pred, daily_pred, status_message).
    """
    scaler_path = root / "results" / "ramt_scaler.joblib"
    y_scaler_path = root / "results" / "ramt_y_scaler.joblib"
    state_path = root / "results" / "ramt_model_state.pt"
    if not scaler_path.is_file() or not state_path.is_file():
        return None, None, "missing_artifacts"

    built = build_live_ramt_sequence(ticker_stem, root, macro_pack)
    if built is None:
        return None, None, "no_sequence"
    seq_unscaled, _reg_parquet = built
    seq_unscaled = apply_predictor_form_to_last_row(
        seq_unscaled, ret_1d_pct, mom_20_pct, rsi, bb_dist, vol_surge
    )

    n_feat_expected = len(ALL_FEATURE_COLS)
    if seq_unscaled.shape[1] != n_feat_expected:
        return None, None, "bad_sequence_shape"

    scaler = joblib.load(scaler_path)
    nf = getattr(scaler, "n_features_in_", None)
    if nf is not None and int(nf) != n_feat_expected:
        return None, None, "scaler_mismatch"

    seq_unscaled = impute_missing_with_training_medians(seq_unscaled, scaler)

    try:
        X = scaler.transform(seq_unscaled).astype(np.float32)
    except ValueError as e:
        err = str(e).lower()
        if "feature" in err or "shape" in err:
            return None, None, "scaler_mismatch"
        raise
    X_t = torch.from_numpy(X).unsqueeze(0)
    regime = int(np.clip(regime_id, 0, 2))
    r = torch.tensor([regime], dtype=torch.long)
    tid = torch.tensor([int(TICKER_TO_ID.get(ticker_stem, 0))], dtype=torch.long)

    try:
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(state_path, map_location="cpu")
    cfg = payload.get("config") or {"seq_len": 30}
    model = build_ramt(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        pred_m, pred_d, _gw = model(X_t, r, ticker_id=tid)

    pm = float(pred_m.squeeze().cpu().numpy())
    pd_ = float(pred_d.squeeze().cpu().numpy())

    if y_scaler_path.is_file():
        try:
            y_scaler = joblib.load(y_scaler_path)
            pm = float(y_scaler.inverse_transform(np.array([[pm]]))[0, 0])
        except Exception:
            pass

    return pm, pd_, "ok"


def render_live_predictor_section(state: SidebarState) -> None:
    stock_snapshot = compute_live_stock_feature_snapshot(state.ticker_code, ROOT)
    if st.session_state.get("lp_seed_ticker") != state.ticker_code:
        if stock_snapshot is not None:
            feats = stock_snapshot["features"]
            st.session_state["lp_ret_1d_pct"] = _log_return_to_pct(float(feats["Ret_1d"]))
            st.session_state["lp_mom_20_pct"] = _log_return_to_pct(float(feats["Ret_21d"]))
            st.session_state["lp_rsi"] = float(feats["RSI_14"])
            st.session_state["lp_bb"] = float(feats["BB_Dist"])
            st.session_state["lp_vol"] = float(feats["Volume_Surge"])
            st.session_state["lp_stock_feature_date"] = str(stock_snapshot["as_of"].date())
            st.session_state["lp_stock_feature_source"] = str(stock_snapshot["source"])
        latest_regime = latest_stock_regime(state.ticker_code, ROOT)
        if latest_regime is not None:
            st.session_state["lp_regime"] = latest_regime
        st.session_state.pop("lp_macro_pack", None)
        st.session_state.pop("lp_fetch_utc", None)
        st.session_state["lp_seed_ticker"] = state.ticker_code

    st.markdown("---")
    st.markdown("### Live predictor")
    st.caption(
        "Stock price/technical slots come from the ticker’s latest raw bar history; macro "
        "slots come from **MarketScraper** with Parquet fallbacks. The **HMM regime** is "
        "passed separately (not in X), and missing values use training medians from "
        "`results/ramt_scaler.joblib` before `transform`."
    )

    b1, b2 = st.columns([1, 2])
    with b1:
        if st.button("Sync live inputs", key="btn_autofill_market"):
            with st.spinner("Syncing live macro pulse and stock technicals…"):
                pack = fetch_live_macro_data_cached(state.ticker_code, str(ROOT))
            st.session_state["lp_macro_pack"] = pack
            st.session_state["lp_fetch_utc"] = datetime.now(timezone.utc).isoformat()
            stock_snapshot = compute_live_stock_feature_snapshot(state.ticker_code, ROOT)
            if stock_snapshot is not None:
                feats = stock_snapshot["features"]
                st.session_state["lp_ret_1d_pct"] = _log_return_to_pct(float(feats["Ret_1d"]))
                st.session_state["lp_mom_20_pct"] = _log_return_to_pct(float(feats["Ret_21d"]))
                st.session_state["lp_rsi"] = float(feats["RSI_14"])
                st.session_state["lp_bb"] = float(feats["BB_Dist"])
                st.session_state["lp_vol"] = float(feats["Volume_Surge"])
                st.session_state["lp_stock_feature_date"] = str(stock_snapshot["as_of"].date())
                st.session_state["lp_stock_feature_source"] = str(stock_snapshot["source"])
            latest_regime = latest_stock_regime(state.ticker_code, ROOT)
            if latest_regime is not None:
                st.session_state["lp_regime"] = latest_regime
            if pack.get("ok"):
                st.session_state["lp_vix"] = float(pack.get("vix_level", float("nan")))
                st.session_state["lp_crude"] = float(pack.get("crude_level", float("nan")))
                st.session_state["lp_inr"] = float(pack.get("inr_level", float("nan")))
            st.rerun()

    with b2:
        fu = st.session_state.get("lp_fetch_utc")
        if fu:
            st.caption(f"Last updated (UTC): **{fu}** · cache TTL 1h")
        else:
            st.caption("Last updated: — (click auto-fill)")

    stock_feature_date = st.session_state.get("lp_stock_feature_date")
    stock_feature_source = st.session_state.get("lp_stock_feature_source", "processed")
    if stock_feature_date:
        st.caption(
            f"Stock technicals for `{state.ticker_code}` sourced from **{stock_feature_source}** "
            f"history as of **{stock_feature_date}**."
        )

    pack = st.session_state.get("lp_macro_pack")
    if pack and pack.get("ok") and pack.get("stale"):
        st.warning(
            "Data is **last available** (not real-time) — market may be closed or the feed delayed."
        )
        d0 = pack.get("latest_data_date")
        if d0 is not None:
            st.caption(f"Latest bar across series: **{pd.Timestamp(d0).date()}**")
    elif pack and not pack.get("ok"):
        st.warning(
            pack.get("error") or "Market data fetch failed — using manual inputs only."
        )

    st.caption(
        f"Spot checks · VIX: `{st.session_state.get('lp_vix', '—')}` · "
        f"Crude: `{st.session_state.get('lp_crude', '—')}` · "
        f"INR: `{st.session_state.get('lp_inr', '—')}`"
    )

    with st.form("live_predictor_form"):
        c1, c2 = st.columns(2)
        with c1:
            ret_pct = st.number_input(
                "Latest stock return (%)",
                value=float(st.session_state.get("lp_ret_1d_pct", 0.0)),
                format="%.4f",
                help="Mapped to the stock’s `Ret_1d` slot after log transform.",
            )
            mom_pct = st.number_input(
                "Latest stock 20-day momentum (%)",
                value=float(st.session_state.get("lp_mom_20_pct", 0.0)),
                format="%.4f",
                help="Mapped to the stock’s `Ret_21d` slot (≈21 trading days).",
            )
            rsi = st.number_input(
                "RSI (14) — stock",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("lp_rsi", 50.0)),
            )
        with c2:
            bb = st.number_input(
                "BB distance (20d) — stock",
                value=float(st.session_state.get("lp_bb", 0.0)),
                format="%.4f",
            )
            vol = st.number_input(
                "Volume surge — stock",
                min_value=0.01,
                max_value=10.0,
                value=float(st.session_state.get("lp_vol", 1.0)),
            )
            regime = st.selectbox(
                "HMM regime",
                options=[0, 1, 2],
                format_func=lambda x: REGIME_NAMES[x],
                index=int(st.session_state.get("lp_regime", 1)),
            )

        go_btn = st.form_submit_button("Run RAMT (scaled inference)", width="stretch")

    macro_for_predict = st.session_state.get("lp_macro_pack") or {}
    if go_btn:
        if not macro_for_predict.get("ok"):
            st.info(
                "Macro columns will use **Parquet fallbacks** until a successful auto-fill "
                "(live Macro_* need a successful **Auto-fill** scrape)."
            )
        with st.spinner("Running RobustScaler + model.forward…"):
            pm, pd_, status = run_ramt_live_predict(
                state.ticker_code,
                ROOT,
                macro_for_predict,
                ret_pct,
                mom_pct,
                rsi,
                bb,
                vol,
                int(regime),
            )
        if status == "missing_artifacts":
            st.info(
                "Train RAMT to create `results/ramt_model_state.pt` and "
                "`results/ramt_scaler.joblib` (e.g. `python models/run_final_2024_2026.py`)."
            )
        elif status == "no_sequence":
            st.warning(f"Could not build a sequence for `{state.ticker_code}` — check Parquet.")
        elif status == "bad_sequence_shape":
            st.error("Internal error: feature matrix width does not match `ALL_FEATURE_COLS`.")
        elif status == "scaler_mismatch":
            st.error(
                "`results/ramt_scaler.joblib` does not match this RAMT build: on disk it encodes a "
                "different **number of features** (often a stale `StandardScaler` from another baseline). "
                f"The live predictor needs **{len(ALL_FEATURE_COLS)}** columns (`ALL_FEATURE_COLS`) from "
                "the **RobustScaler** written by `models/ramt/train_ranking.py` or "
                "`python models/run_final_2024_2026.py`. Delete the bad file and re-run training."
            )
        elif status == "ok" and pm is not None:
            st.success(
                f"**Monthly head (α-space):** {pm:.6f} · **Daily head:** {pd_:.6f} · "
                f"**Regime:** {REGIME_NAMES.get(int(regime), regime)}"
            )
            st.caption(
                "Monthly head matches ranking training target (excess vs NIFTY); "
                "inverse y-scaler applied when `ramt_y_scaler.joblib` exists."
            )
        else:
            st.error("Inference failed unexpectedly.")


# ─── Tab content ─────────────────────────────────────────────
def render_live_performance_tab(
    data: dict[str, Any],
    state: SidebarState,
    ranking_df: pd.DataFrame | None,
    ranking_slice: pd.DataFrame,
    full_slice: pd.DataFrame,
    sim_window: dict[str, Any] | None,
    sim_full: dict[str, Any] | None,
    metrics_win: dict[str, float],
    metrics_full: dict[str, float],
    features_df: pd.DataFrame | None,
    regime_info: dict[str, Any] | None,
    shadow_df: pd.DataFrame,
    shadow_regime: int | None,
) -> None:
    st.markdown("### Live performance")
    st.caption(
        "Window metrics recompute from the selected dates; deltas compare to the full "
        "available backtest (same test-only filter)."
    )

    c1, c2, c3 = st.columns(3)
    da_d = metrics_win["DA%"] - metrics_full["DA%"]
    sh_d = metrics_win["Sharpe"] - metrics_full["Sharpe"]
    md_d = metrics_win["MaxDD"] - metrics_full["MaxDD"]
    c1.metric(
        "Directional accuracy",
        f"{metrics_win['DA%']:.2f}%",
        delta=f"{da_d:+.2f} pp vs full",
        delta_color="normal",
    )
    c2.metric(
        "Sharpe (rebalance α)",
        f"{metrics_win['Sharpe']:.3f}",
        delta=f"{sh_d:+.3f} vs full",
    )
    c3.metric(
        "Max drawdown",
        f"{metrics_win['MaxDD']*100:.2f}%",
        delta=f"{md_d*100:+.2f} pp vs full",
    )

    st.markdown("---")
    if shadow_regime == 2:
        st.markdown(
            '<div class="shadow-badge-bear">Status: PROTECTIVE CASH (0%)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "HMM **BEAR** vetoes live deployment to cash — shadow table still shows model "
            "rankings so the Transformer signal remains visible."
        )

    st.markdown("#### Current model conviction (shadow picks)")
    st.caption(
        "Top 5 tickers by **predicted_alpha** at the as-of date (informational only)."
    )

    if shadow_df is not None and not shadow_df.empty:
        disp = shadow_df.copy()
        disp["predicted_alpha"] = (disp["predicted_alpha"] * 100).map(lambda x: f"{x:.3f}%")
        disp["actual_alpha"] = (disp["actual_alpha"] * 100).map(lambda x: f"{x:.3f}%")
        st.dataframe(disp, width="stretch", hide_index=True)
    else:
        st.info("No ranking rows for this as-of date.")

    st.markdown("---")
    rc1, rc2 = st.columns(2)
    with rc1:
        if regime_info:
            st.metric("Latest HMM regime (ticker file)", regime_info["name"])
    with rc2:
        if shadow_regime is not None:
            st.metric("Regime at as-of (NIFTY)", REGIME_NAMES.get(shadow_regime, "?"))

    st.markdown("---")
    st.markdown("### Equity curve")
    render_charts(sim_window, data.get("regime_series"))

    if sim_window and sim_window.get("rows_detail") is not None and not sim_window["rows_detail"].empty:
        with st.expander("Rebalance detail (window)"):
            rd = sim_window["rows_detail"].drop(columns=["tickers_list"], errors="ignore")
            st.dataframe(rd, width="stretch", hide_index=True)

    render_live_predictor_section(state)

    st.markdown("#### Saved backtest (`run_final` daily engine)")
    bt_df = data.get("backtest_results")
    if bt_df is None or bt_df.empty:
        st.caption("Optional: run `run_final` to populate `results/backtest_results.csv`.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bt_df["date"],
                y=bt_df.get("portfolio_value", (1 + bt_df["cumulative_return"]) * 100000),
                name="Portfolio ₹",
                line=dict(color="#4ade80", width=2),
            )
        )
        if "nifty_value" in bt_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=bt_df["date"],
                    y=bt_df["nifty_value"],
                    name="NIFTY ₹",
                    line=dict(color="#38bdf8", width=2, dash="dash"),
                )
            )
        fig.update_layout(
            paper_bgcolor="#0b1020",
            plot_bgcolor="#111827",
            font=dict(color="#e2e8f0"),
            height=360,
        )
        fig.update_xaxes(gridcolor="#1e293b")
        fig.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig, width="stretch")


def render_training_analytics_tab(
    state: SidebarState,
    ranking_preds_df: pd.DataFrame | None,
    ranking_slice: pd.DataFrame,
    all_predictions: dict[str, pd.DataFrame],
    features_df: pd.DataFrame | None,
) -> None:
    train_png = ROOT / "results" / "training_dashboard.png"
    train_csv = ROOT / "results" / "training_history.csv"
    if train_png.is_file():
        st.markdown("### Training run")
        st.caption("Blind-split RAMT — loss, LR, generalization gap.")
        st.image(str(train_png), width="stretch")
        if train_csv.is_file():
            st.download_button(
                label="Download training_history.csv",
                data=train_csv.read_bytes(),
                file_name="training_history.csv",
                mime="text/csv",
            )
    else:
        st.info("No `results/training_dashboard.png` — run training to populate.")

    st.markdown("---")
    st.markdown("### RAMT predicted vs actual (ranking α)")
    fig_ramt, ramt_m = plot_ramt_ranking_predicted_vs_actual(
        ranking_slice if ranking_slice is not None and not ranking_slice.empty else ranking_preds_df,
        state.ticker_code,
    )
    if fig_ramt is None:
        st.warning("No ranking predictions for this ticker in the current slice.")
    else:
        mcols = st.columns(5)
        mcols[0].metric("DA%", f"{ramt_m.get('DA%', 0):.2f}")
        mcols[1].metric("RMSE", f"{ramt_m.get('RMSE', 0):.5f}")
        mcols[2].metric("MAE", f"{ramt_m.get('MAE', 0):.5f}")
        mcols[3].metric("ρ", f"{ramt_m.get('ρ', 0):.3f}")
        mcols[4].metric("n", f"{ramt_m.get('n', 0)}")
        st.plotly_chart(fig_ramt, width="stretch")

    st.markdown("---")
    st.markdown("### Model comparison")
    fig_comp = plot_metrics_comparison(all_predictions, state.ticker_code)
    if fig_comp is not None:
        st.plotly_chart(fig_comp, width="stretch")
        if all_predictions:
            rows = []
            for model_name, pred_df in all_predictions.items():
                t_df = pred_df[pred_df["Ticker"] == state.ticker_code]
                if t_df.empty:
                    continue
                m = compute_metrics(t_df["y_true"].values, t_df["y_pred"].values)
                m["Model"] = model_name
                rows.append(m)
            if rows:
                mt = pd.DataFrame(rows).set_index("Model")
                st.dataframe(
                    mt.style.format(
                        {
                            "RMSE": "{:.4f}",
                            "MAE": "{:.4f}",
                            "DA%": "{:.2f}%",
                            "Sharpe": "{:.2f}",
                            "MaxDD": "{:.4f}",
                        }
                    ),
                    width="stretch",
                )

    st.markdown("---")
    st.markdown("### Ticker charts")
    primary_preds = load_predictions(state.selected_model.lower())
    c1, c2 = st.columns(2)
    with c1:
        f1 = plot_actual_vs_predicted(primary_preds, state.ticker_code, state.chart_days)
        if f1:
            st.plotly_chart(f1, width="stretch")
    with c2:
        f2 = plot_cumulative_returns(primary_preds, state.ticker_code)
        if f2:
            st.plotly_chart(f2, width="stretch")
    f3 = plot_regime_history(features_df, days=min(180, state.chart_days + 60))
    if f3:
        st.plotly_chart(f3, width="stretch")


def render_raw_data_explorer_tab(
    state: SidebarState,
    primary_preds: pd.DataFrame | None,
    ranking_preds_df: pd.DataFrame | None,
) -> None:
    st.markdown("### Raw predictions")
    if primary_preds is not None and {"y_true", "y_pred"}.issubset(primary_preds.columns):
        t_df = (
            primary_preds[primary_preds["Ticker"] == state.ticker_code]
            .tail(200)
            .sort_values("Date", ascending=False)
        )
        t_df = t_df.copy()
        t_df["Direction_Correct"] = np.sign(t_df["y_true"]) == np.sign(t_df["y_pred"])
        t_df["y_true_pct"] = (t_df["y_true"] * 100).round(4)
        t_df["y_pred_pct"] = (t_df["y_pred"] * 100).round(4)
        st.dataframe(
            t_df[["Date", "y_true_pct", "y_pred_pct", "Direction_Correct"]].rename(
                columns={
                    "y_true_pct": "Actual %",
                    "y_pred_pct": "Pred %",
                    "Direction_Correct": "Dir match",
                }
            ),
            width="stretch",
            height=420,
        )
    elif ranking_preds_df is not None and not ranking_preds_df.empty:
        t_df = ranking_preds_df[ranking_preds_df["Ticker"] == state.ticker_code].tail(200)
        t_df = t_df.sort_values("Date", ascending=False)
        if t_df.empty:
            st.info("No rows for this ticker.")
        else:
            t_df = t_df.copy()
            t_df["Direction_Correct"] = (
                np.sign(t_df["actual_alpha"]) == np.sign(t_df["predicted_alpha"])
            )
            t_df["actual_pct"] = (t_df["actual_alpha"] * 100).round(4)
            t_df["pred_pct"] = (t_df["predicted_alpha"] * 100).round(4)
            st.dataframe(
                t_df[["Date", "actual_pct", "pred_pct", "Direction_Correct"]].rename(
                    columns={
                        "actual_pct": "Actual %",
                        "pred_pct": "Pred %",
                        "Direction_Correct": "Dir match",
                    }
                ),
                width="stretch",
                height=420,
            )
    else:
        st.info("No prediction CSV found under `results/`.")
    st.caption("Tip: use the sidebar refresh if you regenerate results on disk.")


# ─── Main ────────────────────────────────────────────────────
def main() -> None:
    with st.spinner("Ensuring NIFTY index features & loading results…"):
        data = load_data(str(ROOT))
    ranking_df = data["ranking_df"]

    state = render_sidebar(ranking_df)

    nifty_prices = data.get("nifty_prices")

    # Full-sample baseline (same test filter) for metric deltas
    if ranking_df is not None and not ranking_df.empty:
        r = ranking_df.copy()
        r["Date"] = pd.to_datetime(r["Date"])
        d_lo, d_hi = r["Date"].min(), r["Date"].max()
    else:
        d_lo = d_hi = state.date_end

    full_slice = (
        filter_ranking_by_dates(ranking_df, d_lo, d_hi, state.live_test_only)
        if ranking_df is not None
        else pd.DataFrame()
    )
    ranking_slice = filter_ranking_by_dates(
        ranking_df, state.date_start, state.date_end, state.live_test_only
    )

    sim_window = (
        simulate_portfolio(
            ranking_df,
            state.date_start,
            state.date_end,
            state.sim_top_n,
            state.sim_capital,
            state.live_test_only,
            nifty_prices,
            data.get("regime_series"),
        )
        if ranking_df is not None
        else None
    )
    sim_full = (
        simulate_portfolio(
            ranking_df,
            d_lo,
            d_hi,
            state.sim_top_n,
            state.sim_capital,
            state.live_test_only,
            nifty_prices,
            data.get("regime_series"),
        )
        if ranking_df is not None
        else None
    )

    metrics_win = calculate_metrics(ranking_slice, sim_window)
    metrics_full = calculate_metrics(full_slice, sim_full)

    features_df = load_features(state.ticker_code)
    regime_info = get_current_regime(features_df)
    reg_series = data.get("regime_series")
    shadow_regime = regime_at(reg_series, state.as_of_date)
    shadow_df = (
        top5_shadow_for_date(
            ranking_df,
            state.as_of_date,
        )
        if ranking_df is not None
        else pd.DataFrame()
    )

    all_predictions = load_all_predictions()

    st.markdown("# RAMT · production terminal")
    st.caption("Live simulator · shadow conviction · market scraper · JetBrains Mono")

    tab_live, tab_train, tab_raw = st.tabs(["Live performance", "Training analytics", "Raw data explorer"])

    with tab_live:
        render_live_performance_tab(
            data,
            state,
            ranking_df,
            ranking_slice,
            full_slice,
            sim_window,
            sim_full,
            metrics_win,
            metrics_full,
            features_df,
            regime_info,
            shadow_df,
            shadow_regime,
        )

    with tab_train:
        render_training_analytics_tab(
            state,
            ranking_df,
            ranking_slice,
            all_predictions,
            features_df,
        )

    with tab_raw:
        render_raw_data_explorer_tab(
            state,
            load_predictions(state.selected_model.lower()),
            ranking_df,
        )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#475569;font-size:12px;'>"
        "RAMT — research only · not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
