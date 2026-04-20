"""
Momentum + Regime + Sector — NIFTY 200 research dashboard.

Loads all strategy metrics from ``results/final_strategy/backtest_results.csv`` and benchmarks
from ``data/raw/_NSEI.parquet``. No live model inference.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.feature_engineering import _safe_stem_from_ticker  # noqa: E402
from features.sectors import get_sector  # noqa: E402

BACKTEST_CSV = ROOT / "results" / "final_strategy" / "backtest_results.csv"
WEEKLY_BT_CSV = ROOT / "results" / "final_strategy" / "backtest_results_weekly_2023_2026.csv"
WEEKLY_RET5D_BT_CSV = (
    ROOT / "results" / "final_strategy" / "backtest_results_weekly_ret5d_2023_2026.csv"
)
NIFTY_PARQUET = ROOT / "data" / "raw" / "_NSEI.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"

# Optional archived comparison CSVs (if present)
ARCHIVE_RAMT_BACKTEST = ROOT / "results" / "archive" / "ramt_backtest_results.csv"
ARCHIVE_MOM_NO_SECTOR = ROOT / "results" / "archive" / "momentum_regime_no_sector_backtest.csv"

# Phase 1 / Phase 2 ML exports (optional — may be absent)
# Walk-forward baselines (``models/baseline_xgboost.py`` / ``baseline_lstm.py`` → this folder)
BASELINE_WALKFORWARD = ROOT / "results" / "phase1_baselines"
PHASE1_DAILY = BASELINE_WALKFORWARD
PHASE2_MONTHLY = ROOT / "results" / "phase2_monthly"
RAMT_DIR = ROOT / "results" / "ramt"

REGIME_FILL = {
    "BULL": "rgba(34, 197, 94, 0.16)",
    "HIGH_VOL": "rgba(250, 204, 21, 0.22)",
    "BEAR": "rgba(248, 113, 113, 0.18)",
}

REGIME_LINE = {
    "BULL": "#22c55e",
    "HIGH_VOL": "#eab308",
    "BEAR": "#ef4444",
}


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
    h1, h2, h3 {{
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: #38bdf8 !important;
    }}
    .metric-caption {{
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Momentum + Regime · NIFTY 200",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_theme_css()


def compute_metrics_with_windows_per_year(
    bt_path: str | Path,
    *,
    windows_per_year: float,
    capital: float = 100_000,
) -> dict[str, Any]:
    """
    Compute the same metrics as `compute_metrics`, but with explicit Sharpe annualization.

    This is used for *separate experiments* (e.g. weekly) without changing the dashboard's
    historical monthly assumptions for the production strategy.
    """
    bt = pd.read_csv(bt_path, parse_dates=["date"])
    r = bt["portfolio_return"].dropna()
    nav = bt["portfolio_value"].values.astype(float)
    start_ts = bt["date"].iloc[0]
    end_ts = bt["date"].iloc[-1]
    span_years = (end_ts - start_ts).days / 365.25

    wpy = float(windows_per_year)
    sharpe_net = float(r.mean() / r.std() * np.sqrt(wpy)) if r.std() > 0 else 0.0

    total_ret = nav[-1] / capital - 1.0
    cagr = (1 + total_ret) ** (1 / span_years) - 1 if span_years > 0 else 0.0

    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())

    win_rate = float((r > 0).mean())

    return {
        "sharpe_net": sharpe_net,
        "cagr": float(cagr),
        "max_dd": max_dd,
        "win_rate": win_rate,
        "total_return": float(total_ret),
        "final_nav": float(nav[-1]),
        "n_windows": len(bt),
        "span_years": float(span_years),
    }


def compute_metrics(bt_path: str | Path, capital: float = 100_000) -> dict[str, Any]:
    bt = pd.read_csv(bt_path, parse_dates=["date"])
    r = bt["portfolio_return"].dropna()
    nav = bt["portfolio_value"].values.astype(float)
    start_ts = bt["date"].iloc[0]
    end_ts = bt["date"].iloc[-1]
    span_years = (end_ts - start_ts).days / 365.25

    # Historical dashboard assumption: these backtests are monthly-ish windows.
    sharpe_net = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0

    total_ret = nav[-1] / capital - 1.0
    cagr = (1 + total_ret) ** (1 / span_years) - 1 if span_years > 0 else 0.0

    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())

    win_rate = float((r > 0).mean())

    return {
        "sharpe_net": sharpe_net,
        "cagr": float(cagr),
        "max_dd": max_dd,
        "win_rate": win_rate,
        "total_return": float(total_ret),
        "final_nav": float(nav[-1]),
        "n_windows": len(bt),
        "span_years": float(span_years),
    }


def compute_nifty_benchmark(
    nifty_parquet: str | Path, start: Any, end: Any, capital: float = 100_000
) -> dict[str, Any]:
    n = pd.read_parquet(nifty_parquet)
    n["Date"] = pd.to_datetime(n["Date"])
    n = n[(n["Date"] >= start) & (n["Date"] <= end)].sort_values("Date")
    if n.empty:
        raise ValueError("NIFTY series empty for requested range.")
    px = n["Adj Close"].astype(float).values
    nav = capital * px / px[0]
    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())
    total_ret = nav[-1] / capital - 1.0
    span_years = (n["Date"].iloc[-1] - n["Date"].iloc[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / span_years) - 1 if span_years > 0 else 0.0
    daily_ret = pd.Series(px).pct_change().dropna()
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    )
    return {
        "cagr": float(cagr),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_return": float(total_ret),
        "nav_series": pd.DataFrame({"date": n["Date"].values, "nav": nav}),
        "adj_close": n.set_index("Date")["Adj Close"].astype(float),
    }


@st.cache_data(show_spinner=False)
def load_backtest_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_nifty_prices(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    n = pd.read_parquet(p)
    n["Date"] = pd.to_datetime(n["Date"])
    return n.sort_values("Date")


def nifty_nav_at_rebalance_dates(
    bt: pd.DataFrame, nifty: pd.DataFrame, capital: float = 100_000
) -> pd.DataFrame:
    """Buy-and-hold NIFTY NAV at each strategy rebalance date (₹ start = capital)."""
    s = nifty.sort_values("Date").copy()
    dates = pd.to_datetime(bt["date"]).sort_values()
    out_nav: list[float] = []
    out_px: list[float] = []
    for d in dates:
        ts = pd.Timestamp(d)
        sub = s[s["Date"] <= ts]
        if sub.empty:
            out_nav.append(float("nan"))
            out_px.append(float("nan"))
            continue
        p0 = float(sub["Adj Close"].iloc[-1])
        out_px.append(p0)
    p_start = out_px[0]
    for p in out_px:
        out_nav.append(capital * (p / p_start) if p == p and p_start > 0 else float("nan"))
    return pd.DataFrame({"date": dates.values, "nifty_nav": out_nav, "nifty_px": out_px})


def nifty_inter_rebalance_win_rate(bt: pd.DataFrame, nifty: pd.DataFrame) -> float:
    """Share of positive NIFTY returns between consecutive rebalance dates."""
    s = nifty.sort_values("Date")
    dates = pd.to_datetime(bt["date"]).sort_values()
    rets: list[float] = []
    for i in range(1, len(dates)):
        sub_i = s[s["Date"] <= pd.Timestamp(dates.iloc[i])]
        sub_j = s[s["Date"] <= pd.Timestamp(dates.iloc[i - 1])]
        if sub_i.empty or sub_j.empty:
            continue
        pi = float(sub_i["Adj Close"].iloc[-1])
        pj = float(sub_j["Adj Close"].iloc[-1])
        rets.append(pi / pj - 1.0)
    if not rets:
        return 0.0
    return float(np.mean(np.array(rets) > 0))


def add_regime_vrects(fig: go.Figure, bt: pd.DataFrame) -> None:
    dts = pd.to_datetime(bt["date"]).tolist()
    regimes = bt["regime"].astype(str).tolist()
    for i in range(len(dts) - 1):
        reg = regimes[i]
        fill = REGIME_FILL.get(reg, "rgba(100,116,139,0.12)")
        fig.add_vrect(
            x0=dts[i],
            x1=dts[i + 1],
            fillcolor=fill,
            layer="below",
            line_width=0,
        )
    if dts:
        last = dts[-1]
        reg = regimes[-1]
        fill = REGIME_FILL.get(reg, "rgba(100,116,139,0.12)")
        fig.add_vrect(
            x0=last,
            x1=last + pd.Timedelta(days=2),
            fillcolor=fill,
            layer="below",
            line_width=0,
        )


def parse_stocks_held(cell: Any) -> list[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    s = str(cell).strip()
    try:
        return [str(x) for x in ast.literal_eval(s)]
    except (ValueError, SyntaxError):
        return []


@st.cache_data(show_spinner=False)
def feature_row_at_date(stem: str, asof_ns: int, processed_s: str) -> pd.DataFrame | None:
    processed = Path(processed_s)
    path = processed / f"{stem}_features.parquet"
    if not path.is_file():
        return None
    asof = pd.Timestamp(asof_ns)
    df = pd.read_parquet(path, columns=["Date", "Ret_21d", "Sector_Alpha", "Monthly_Alpha"])
    df["Date"] = pd.to_datetime(df["Date"])
    sub = df[df["Date"] <= asof]
    if sub.empty:
        return None
    row = sub.iloc[-1:].copy()
    row["stem"] = stem
    return row


def optional_metrics_from_csv(path: Path, capital: float = 100_000) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return compute_metrics(path, capital=capital)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_pred_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """Map common column names to predicted / actual."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lc = {c.lower(): c for c in out.columns}
    pred_c = None
    for key in ("predicted_alpha", "predicted", "pred", "y_hat", "prediction"):
        if key in lc:
            pred_c = lc[key]
            break
    act_c = None
    for key in ("actual_alpha", "actual", "y", "target"):
        if key in lc:
            act_c = lc[key]
            break
    if pred_c is None or act_c is None:
        return None
    out = out.rename(columns={pred_c: "predicted", act_c: "actual"})
    return out


def _mean_cross_sectional_ic(df: pd.DataFrame) -> float:
    if "Date" not in df.columns:
        return float("nan")
    ics: list[float] = []
    for _, g in df.groupby("Date"):
        if len(g) < 4:
            continue
        ic = g["predicted"].corr(g["actual"], method="spearman")
        if ic == ic:
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else float("nan")


def _directional_accuracy(pred: np.ndarray, act: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(act, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p * a) > 0))


def _plotly_dark() -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "#0b1020",
        "plot_bgcolor": "#0f172a",
    }


def render_ramt_transformer_section() -> None:
    """RAMT: metrics + equity + scatter + shadow picks + training PNG from ``results/ramt/``."""
    st.subheader("RAMT transformer — Phase 2 (monthly alpha)")
    st.caption(
        "Regime-adaptive multimodal transformer; blind-test metrics and backtest from "
        f"`{RAMT_DIR.relative_to(ROOT)}/` (no live inference)."
    )

    mpath = RAMT_DIR / "ramt_metrics.json"
    bt_ramt = RAMT_DIR / "backtest_results.csv"
    rank_path = RAMT_DIR / "ranking_predictions.csv"
    train_png = RAMT_DIR / "training_dashboard.png"

    if not mpath.is_file() and not bt_ramt.is_file():
        st.info(
            "N/A — no RAMT metrics or backtest found. Run `scripts/regenerate_ramt_outputs.py` "
            f"or place artifacts under `{RAMT_DIR}`."
        )
        return

    mj = _load_json(mpath) or {}
    strat_ramt = optional_metrics_from_csv(bt_ramt) if bt_ramt.is_file() else None

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        da = mj.get("DA_pct")
        st.metric("Directional accuracy (DA)", f"{float(da):.2f}%" if da is not None else "N/A")
    with c2:
        mic = mj.get("mean_IC")
        st.metric("Mean IC", f"{float(mic):.4f}" if mic is not None else "N/A")
    with c3:
        sh = mj.get("Sharpe")
        st.metric("Sharpe (net)", f"{float(sh):.2f}" if sh is not None else "N/A")
    with c4:
        if strat_ramt:
            st.metric("CAGR", f"{100 * strat_ramt['cagr']:.1f}%")
        else:
            st.metric("CAGR", "N/A")
    with c5:
        mdd = mj.get("MaxDD")
        if mdd is not None:
            st.metric("Max drawdown", f"{100 * float(mdd):.1f}%")
        elif strat_ramt:
            st.metric("Max drawdown", f"{100 * strat_ramt['max_dd']:.1f}%")
        else:
            st.metric("Max drawdown", "N/A")

    st.caption(
        "RMSE / MAE (blind test): "
        f"{mj.get('RMSE', 'N/A')} / {mj.get('MAE', 'N/A')} — from `ramt_metrics.json` when present."
    )

    if bt_ramt.is_file() and NIFTY_PARQUET.is_file():
        try:
            bt = load_backtest_csv(str(bt_ramt))
            nifty_raw = load_nifty_prices(str(NIFTY_PARQUET))
            nav_df = nifty_nav_at_rebalance_dates(bt, nifty_raw)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bt["date"],
                    y=bt["portfolio_value"],
                    name="RAMT strategy NAV",
                    line=dict(color="#a78bfa", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=nav_df["date"],
                    y=nav_df["nifty_nav"],
                    name="NIFTY buy-and-hold",
                    line=dict(color="#94a3b8", width=2, dash="dot"),
                )
            )
            add_regime_vrects(fig, bt)
            fig.update_layout(
                title="RAMT — equity vs NIFTY (₹100k start); regime shading",
                xaxis_title="Date",
                yaxis_title="NAV (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60),
                **_plotly_dark(),
            )
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not plot RAMT equity curve: {e}")
    else:
        st.info("Equity curve: add `results/ramt/backtest_results.csv` and NIFTY raw parquet.")

    if rank_path.is_file():
        try:
            rdf = pd.read_csv(rank_path)
            rdf["Date"] = pd.to_datetime(rdf["Date"])
            norm = _normalize_pred_df(rdf)
            if norm is not None:
                if "Period" in rdf.columns:
                    sub = norm[rdf["Period"].astype(str) == "Test"].copy()
                else:
                    sub = norm.copy()
                sub = sub.dropna(subset=["predicted", "actual"])
                if len(sub) > 0:
                    samp = sub.sample(min(4000, len(sub)), random_state=0) if len(sub) > 4000 else sub
                    fig_sc = go.Figure(
                        data=go.Scatter(
                            x=samp["actual"],
                            y=samp["predicted"],
                            mode="markers",
                            marker=dict(size=5, color="#38bdf8", opacity=0.35),
                            name="pred vs actual",
                        )
                    )
                    mn = float(min(samp["actual"].min(), samp["predicted"].min()))
                    mx = float(max(samp["actual"].max(), samp["predicted"].max()))
                    fig_sc.add_trace(
                        go.Scatter(x=[mn, mx], y=[mn, mx], name="y=x", line=dict(color="#64748b", dash="dash"))
                    )
                    fig_sc.update_layout(
                        title="RAMT — predicted vs actual (blind test sample)",
                        xaxis_title="Actual alpha",
                        yaxis_title="Predicted alpha",
                        **_plotly_dark(),
                    )
                    st.plotly_chart(fig_sc, width="stretch")

            last_d = rdf["Date"].max()
            top = (
                rdf[rdf["Date"] == last_d]
                .sort_values("predicted_alpha", ascending=False)
                .head(8)
            )
            if not top.empty:
                st.subheader("Model conviction (shadow picks — last rebalance date in file)")
                st.dataframe(
                    top[["Date", "Ticker", "predicted_alpha", "actual_alpha"]]
                    if "actual_alpha" in top.columns
                    else top,
                    width="stretch",
                    hide_index=True,
                )
        except Exception as e:
            st.warning(f"Ranking predictions chart/table: {e}")
    else:
        st.info("`ranking_predictions.csv` not found under results/ramt/.")

    if train_png.is_file():
        st.subheader("Training analytics")
        st.image(str(train_png), use_container_width=True)
    else:
        st.caption("No `training_dashboard.png` in results/ramt/.")


def render_phase1_daily_block(
    model_label: str,
    pred_path: Path,
    metrics_path: Path,
) -> None:
    st.markdown(f"### {model_label} — Phase 1 (daily return prediction)")
    st.caption(
        "Original Phase 1 attempt — predicted next-day return on NIFTY 200. Not a ranking signal."
    )
    if not pred_path.is_file():
        st.info(
            f"N/A — not reproducible in this repo: missing `{pred_path.relative_to(ROOT)}`. "
            "No Phase 1 daily predictions CSV found."
        )
        return

    mj = _load_json(metrics_path) if metrics_path.is_file() else None
    df = pd.read_csv(pred_path)
    norm = _normalize_pred_df(df)
    if norm is None:
        st.error(f"Could not find predicted/actual columns in `{pred_path.name}`.")
        return
    if mj is None:
        pred = norm["predicted"].values
        act = norm["actual"].values
        if "Date" in df.columns:
            ic_df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(df["Date"]),
                    "predicted": pred,
                    "actual": act,
                }
            )
            mic = _mean_cross_sectional_ic(ic_df)
        else:
            mic = float("nan")
        mj = {
            "directional_accuracy": _directional_accuracy(pred, act),
            "mean_ic": mic,
            "rmse": float(np.sqrt(np.mean((pred - act) ** 2))),
            "mae": float(np.abs(pred - act).mean()),
        }
        st.caption("Metrics computed on the fly from predictions (metrics JSON was missing).")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Directional accuracy", f"{100 * float(mj.get('directional_accuracy', 0)):.2f}%")
    with c2:
        st.metric("Mean IC", f"{float(mj.get('mean_ic', float('nan'))):.4f}")
    with c3:
        st.metric("RMSE", f"{float(mj.get('rmse', float('nan'))):.6f}")

    st.info(
        "Daily signal-to-noise ratio too low; target was re-specified to monthly alpha in Phase 2."
    )

    if "Date" in df.columns:
        norm_plot = norm.assign(Date=pd.to_datetime(df["Date"]))
    else:
        norm_plot = norm

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=norm_plot["predicted"], name="Predicted", opacity=0.6, nbinsx=50)
    )
    fig.add_trace(go.Histogram(x=norm_plot["actual"], name="Actual", opacity=0.6, nbinsx=50))
    fig.update_layout(
        title="Distribution: predicted vs actual",
        barmode="overlay",
        xaxis_title="Value",
        yaxis_title="Count",
        **_plotly_dark(),
    )
    st.plotly_chart(fig, width="stretch")


def render_phase2_monthly_block(
    model_label: str,
    pred_path: Path,
    metrics_path: Path,
    backtest_path: Path,
    *,
    baseline_callout: bool = False,
) -> None:
    st.markdown(f"### {model_label} — Phase 2 (21-day alpha prediction)")
    if "LSTM" in model_label:
        st.caption(
            "LSTM retrained on monthly alpha target — direct comparison point to the RAMT transformer."
        )
    else:
        st.caption(
            "XGBoost on monthly alpha; Phase 2 gradient-boosting baseline vs RAMT."
        )
    if baseline_callout:
        st.success(
            "Phase 2 XGBoost is the baseline against which RAMT is compared (when metrics exist)."
        )

    if not pred_path.is_file():
        st.info(
            f"N/A — not reproducible: missing `{pred_path.relative_to(ROOT)}`."
        )
        return

    mj = _load_json(metrics_path) if metrics_path.is_file() else None
    df = pd.read_csv(pred_path)
    norm = _normalize_pred_df(df)
    if norm is None:
        st.error(f"Could not resolve prediction columns in `{pred_path.name}`.")
        return

    if mj is None:
        pred = norm["predicted"].values
        act = norm["actual"].values
        has_date = "Date" in df.columns
        if has_date:
            norm_ic = norm.assign(Date=pd.to_datetime(df["Date"]))
        else:
            norm_ic = norm.assign(Date=pd.RangeIndex(len(norm)))
        mj = {
            "directional_accuracy": _directional_accuracy(pred, act),
            "mean_ic": _mean_cross_sectional_ic(norm_ic),
            "rmse": float(np.sqrt(np.mean((pred - act) ** 2))),
            "mae": float(np.abs(pred - act).mean()),
            "top5_positive_rate": float("nan"),
        }
        if has_date:
            t5 = []
            for _, g in norm_ic.groupby("Date"):
                g2 = g.sort_values("predicted", ascending=False).head(5)
                if len(g2):
                    t5.append(float(g2["actual"].mean() > 0))
            mj["top5_positive_rate"] = float(np.mean(t5)) if t5 else float("nan")
        st.caption("Metrics computed from predictions (JSON missing).")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("DA", f"{100 * float(mj.get('directional_accuracy', 0)):.2f}%")
    with c2:
        st.metric("Mean IC", f"{float(mj.get('mean_ic', float('nan'))):.4f}")
    with c3:
        sv = mj.get("sharpe")
        st.metric("Sharpe", f"{float(sv):.2f}" if sv is not None and sv == sv else "N/A")
    with c4:
        cv = mj.get("cagr")
        st.metric("CAGR", f"{100 * float(cv):.1f}%" if cv is not None and cv == cv else "N/A")
    with c5:
        mv = mj.get("max_dd")
        if mv is None:
            mv = mj.get("MaxDD")
        st.metric("Max DD", f"{100 * float(mv):.1f}%" if mv is not None and mv == mv else "N/A")

    if backtest_path.is_file() and NIFTY_PARQUET.is_file():
        try:
            bt = load_backtest_csv(str(backtest_path))
            nifty_raw = load_nifty_prices(str(NIFTY_PARQUET))
            nav_df = nifty_nav_at_rebalance_dates(bt, nifty_raw)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bt["date"],
                    y=bt["portfolio_value"],
                    name=f"{model_label} NAV",
                    line=dict(color="#f472b6", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=nav_df["date"],
                    y=nav_df["nifty_nav"],
                    name="NIFTY",
                    line=dict(color="#94a3b8", width=2, dash="dot"),
                )
            )
            add_regime_vrects(fig, bt)
            fig.update_layout(
                title=f"{model_label} — equity vs NIFTY",
                xaxis_title="Date",
                yaxis_title="NAV (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50),
                **_plotly_dark(),
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Equity plot failed: {e}")
    else:
        st.info("Backtest not yet regenerated for this model (need `*_backtest_results.csv`).")

    if norm is not None and len(norm) > 0:
        sub = norm.dropna(subset=["predicted", "actual"])
        samp = sub.sample(min(3000, len(sub)), random_state=1) if len(sub) > 3000 else sub
        fig2 = go.Figure(
            data=go.Scatter(
                x=samp["actual"],
                y=samp["predicted"],
                mode="markers",
                marker=dict(size=4, color="#f472b6", opacity=0.3),
            )
        )
        mn = float(min(samp["actual"].min(), samp["predicted"].min()))
        mx = float(max(samp["actual"].max(), samp["predicted"].max()))
        fig2.add_trace(
            go.Scatter(x=[mn, mx], y=[mn, mx], line=dict(color="#64748b", dash="dash"), name="y=x")
        )
        fig2.update_layout(
            title="Predicted vs actual (sample)",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            **_plotly_dark(),
        )
        st.plotly_chart(fig2, width="stretch")


def render_momentum_strategy_tabs(
    bt: pd.DataFrame,
    nifty_raw: pd.DataFrame,
    strat: dict[str, Any],
    bench: dict[str, Any],
) -> None:
    """Original dashboard: Strategy Performance / Benchmarks / Positions / Research notes."""
    nifty_win = nifty_inter_rebalance_win_rate(bt, nifty_raw)

    tab_perf, tab_bench, tab_weekly, tab_pos, tab_notes = st.tabs(
        [
            "Strategy Performance",
            "Strategy vs Benchmarks",
            "Weekly experiments",
            "Positions",
            "Research notes",
        ]
    )

    with tab_perf:
        st.subheader("Headline metrics (from `backtest_results.csv`)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Net Sharpe",
                f"{strat['sharpe_net']:.2f}",
                delta=f"{strat['sharpe_net'] - bench['sharpe']:.2f} vs NIFTY",
                help="Annualized from monthly window returns: mean/std × √12",
            )
            st.caption("Sharpe = mean(portfolio_return) / std × √12 (≈12 windows/year)")
        with c2:
            st.metric(
                "CAGR",
                f"{100 * strat['cagr']:.1f}%",
                delta=f"{100 * (strat['cagr'] - bench['cagr']):.1f} pp vs NIFTY",
            )
            st.caption("From first/last `portfolio_value` and calendar span vs initial capital.")
        with c3:
            st.metric(
                "Max drawdown",
                f"{100 * strat['max_dd']:.1f}%",
                delta=f"{100 * (strat['max_dd'] - bench['max_dd']):.1f} pp vs NIFTY",
            )
            st.caption("Peak-to-trrough on `portfolio_value`.")
        with c4:
            st.metric(
                "Win rate",
                f"{100 * strat['win_rate']:.0f}%",
                delta=f"{100 * (strat['win_rate'] - nifty_win):.0f} pp vs NIFTY",
                help="Strategy: share of windows with portfolio_return > 0. "
                "NIFTY: share of positive inter-rebalance returns.",
            )
            st.caption("Win rate (strategy windows vs NIFTY inter-rebalance periods).")

        st.subheader("Equity curve")
        nav_df = nifty_nav_at_rebalance_dates(bt, nifty_raw)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bt["date"],
                y=bt["portfolio_value"],
                name="Strategy NAV",
                line=dict(color="#38bdf8", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=nav_df["date"],
                y=nav_df["nifty_nav"],
                name="NIFTY buy-and-hold (aligned dates)",
                line=dict(color="#94a3b8", width=2, dash="dot"),
            )
        )
        add_regime_vrects(fig, bt)
        fig.update_layout(
            title="Portfolio vs NIFTY (₹100,000 start) — background = regime at rebalance",
            xaxis_title="Date",
            yaxis_title="NAV (₹)",
            template="plotly_dark",
            paper_bgcolor="#0b1020",
            plot_bgcolor="#0f172a",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60),
        )
        fig.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")

        legend_cols = st.columns(3)
        for i, (name, key) in enumerate(
            [("BULL", "BULL"), ("HIGH_VOL", "HIGH_VOL"), ("BEAR", "BEAR")]
        ):
            with legend_cols[i]:
                st.markdown(
                    f'<span style="color:{REGIME_LINE[key]}">■</span> {key.replace("_", " ")}',
                    unsafe_allow_html=True,
                )

        st.subheader("Per-window breakdown")
        disp = bt.copy()
        disp["Stocks Held"] = disp["stocks_held"].map(
            lambda x: ", ".join(parse_stocks_held(x)) if parse_stocks_held(x) else ""
        )
        disp["Portfolio Return (%)"] = disp["portfolio_return"] * 100.0
        disp["Cumulative Return"] = disp["cumulative_return"]
        show = disp[
            [
                "date",
                "regime",
                "Stocks Held",
                "Portfolio Return (%)",
                "turnover",
                "friction_cost",
                "Cumulative Return",
            ]
        ].rename(
            columns={
                "date": "Date",
                "regime": "Regime",
                "turnover": "Turnover",
                "friction_cost": "Friction Cost",
            }
        )
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Portfolio Return (%)": st.column_config.NumberColumn(format="%.2f"),
                "Cumulative Return": st.column_config.NumberColumn(format="%.4f"),
                "Friction Cost": st.column_config.NumberColumn(format="%.2f"),
                "Turnover": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    with tab_bench:
        st.subheader("CAGR, drawdown, Sharpe, win rate")
        ramt_m = optional_metrics_from_csv(ARCHIVE_RAMT_BACKTEST)
        mom_ns_m = optional_metrics_from_csv(ARCHIVE_MOM_NO_SECTOR)
        weekly_m = (
            compute_metrics_with_windows_per_year(WEEKLY_BT_CSV, windows_per_year=52)
            if WEEKLY_BT_CSV.is_file()
            else None
        )
        weekly_ret5d_m = (
            compute_metrics_with_windows_per_year(WEEKLY_RET5D_BT_CSV, windows_per_year=52)
            if WEEKLY_RET5D_BT_CSV.is_file()
            else None
        )

        rows: list[dict[str, str]] = [
            {
                "Strategy": "NIFTY buy-and-hold",
                "CAGR": f"{100 * bench['cagr']:.1f}%",
                "Max DD": f"{100 * bench['max_dd']:.1f}%",
                "Sharpe": f"{bench['sharpe']:.2f}",
                "Win rate": "—",
            },
        ]
        if ramt_m:
            rows.append(
                {
                    "Strategy": "RAMT transformer (archived)",
                    "CAGR": f"{100 * ramt_m['cagr']:.1f}%",
                    "Max DD": f"{100 * ramt_m['max_dd']:.1f}%",
                    "Sharpe": f"{ramt_m['sharpe_net']:.2f}",
                    "Win rate": f"{100 * ramt_m['win_rate']:.0f}%",
                }
            )
        if mom_ns_m:
            rows.append(
                {
                    "Strategy": "Momentum + regime (no sector cap, archived)",
                    "CAGR": f"{100 * mom_ns_m['cagr']:.1f}%",
                    "Max DD": f"{100 * mom_ns_m['max_dd']:.1f}%",
                    "Sharpe": f"{mom_ns_m['sharpe_net']:.2f}",
                    "Win rate": f"{100 * mom_ns_m['win_rate']:.0f}%",
                }
            )
        if weekly_m:
            rows.append(
                {
                    "Strategy": "Momentum + regime + sector (weekly, 2023–2026)",
                    "CAGR": f"{100 * weekly_m['cagr']:.1f}%",
                    "Max DD": f"{100 * weekly_m['max_dd']:.1f}%",
                    "Sharpe": f"{weekly_m['sharpe_net']:.2f}",
                    "Win rate": f"{100 * weekly_m['win_rate']:.0f}%",
                }
            )
        if weekly_ret5d_m:
            rows.append(
                {
                    "Strategy": "Momentum + regime + sector (weekly Ret_5d, 2023–2026)",
                    "CAGR": f"{100 * weekly_ret5d_m['cagr']:.1f}%",
                    "Max DD": f"{100 * weekly_ret5d_m['max_dd']:.1f}%",
                    "Sharpe": f"{weekly_ret5d_m['sharpe_net']:.2f}",
                    "Win rate": f"{100 * weekly_ret5d_m['win_rate']:.0f}%",
                }
            )
        rows.append(
            {
                "Strategy": "Momentum + regime + sector (current)",
                "CAGR": f"{100 * strat['cagr']:.1f}%",
                "Max DD": f"{100 * strat['max_dd']:.1f}%",
                "Sharpe": f"{strat['sharpe_net']:.2f}",
                "Win rate": f"{100 * strat['win_rate']:.0f}%",
            }
        )

        bench_df = pd.DataFrame(rows)
        st.dataframe(bench_df, width="stretch", hide_index=True)
        st.caption(
            "NIFTY metrics are computed from `data/raw/_NSEI.parquet` over the same "
            "calendar span as the strategy. Archived rows appear only if the corresponding "
            "CSV exists under `results/archive/`. Weekly experiment row appears if "
            f"`{WEEKLY_BT_CSV.relative_to(ROOT)}` exists. Weekly Ret_5d row appears if "
            f"`{WEEKLY_RET5D_BT_CSV.relative_to(ROOT)}` exists."
        )

        st.subheader("Monthly returns heatmap (rebalance months)")
        hm = bt.copy()
        hm["year"] = hm["date"].dt.year
        hm["month"] = hm["date"].dt.month
        pivot = hm.pivot_table(index="year", columns="month", values="portfolio_return", aggfunc="first")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.rename(columns={i + 1: month_names[i] for i in range(12) if (i + 1) in pivot.columns})
        z = pivot.values * 100.0
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z,
                x=list(pivot.columns),
                y=pivot.index.astype(str),
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="%"),
                hovertemplate="Year %{y} %{x}<br>Return %{z:.2f}%<extra></extra>",
            )
        )
        fig_hm.update_layout(
            title="Portfolio return by calendar month (% per rebalance row)",
            template="plotly_dark",
            paper_bgcolor="#0b1020",
            plot_bgcolor="#0f172a",
            xaxis_title="Month",
            yaxis_title="Year",
        )
        st.plotly_chart(fig_hm, width="stretch")

        st.subheader("Drawdown (strategy)")
        pv = bt["portfolio_value"].astype(float).values
        peak = np.maximum.accumulate(pv)
        dd = (pv - peak) / peak
        fig_dd = go.Figure(
            data=go.Scatter(
                x=bt["date"],
                y=dd * 100.0,
                fill="tozeroy",
                line=dict(color="#f87171"),
                name="Drawdown",
            )
        )
        fig_dd.update_layout(
            title="Underwater plot — strategy",
            yaxis_title="Drawdown (%)",
            xaxis_title="Date",
            template="plotly_dark",
            paper_bgcolor="#0b1020",
            plot_bgcolor="#0f172a",
        )
        st.plotly_chart(fig_dd, width="stretch")

    with tab_weekly:
        st.subheader("Weekly experiments")
        st.caption(
            "Weekly backtests use the same risk + friction rules as the production strategy, "
            "but on a weekly rebalance grid. Each sub-tab mirrors the production layout: "
            "headline metrics → equity curve."
        )

        def _plot_weekly_equity(bt_path: Path, label: str, color: str) -> None:
            if not bt_path.is_file():
                st.info(f"Missing `{bt_path.relative_to(ROOT)}`.")
                return
            try:
                wbt = load_backtest_csv(str(bt_path))
                w_metrics = compute_metrics_with_windows_per_year(bt_path, windows_per_year=52)
                w_bench = compute_nifty_benchmark(
                    NIFTY_PARQUET, wbt["date"].iloc[0], wbt["date"].iloc[-1]
                )

                st.subheader(f"Headline metrics (from `{bt_path.name}`)")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Net Sharpe",
                        f"{w_metrics['sharpe_net']:.2f}",
                        delta=f"{w_metrics['sharpe_net'] - w_bench['sharpe']:.2f} vs NIFTY",
                        help="Weekly windows annualized as mean/std × √52",
                    )
                with c2:
                    st.metric(
                        "CAGR",
                        f"{100 * w_metrics['cagr']:.1f}%",
                        delta=f"{100 * (w_metrics['cagr'] - w_bench['cagr']):.1f} pp vs NIFTY",
                    )
                with c3:
                    st.metric(
                        "Max drawdown",
                        f"{100 * w_metrics['max_dd']:.1f}%",
                        delta=f"{100 * (w_metrics['max_dd'] - w_bench['max_dd']):.1f} pp vs NIFTY",
                    )
                with c4:
                    # Keep consistent with production: NIFTY win-rate computed on inter-rebalance periods.
                    w_nifty_win = nifty_inter_rebalance_win_rate(wbt, nifty_raw)
                    st.metric(
                        "Win rate",
                        f"{100 * w_metrics['win_rate']:.0f}%",
                        delta=f"{100 * (w_metrics['win_rate'] - w_nifty_win):.0f} pp vs NIFTY",
                    )

                st.subheader("Equity curve")
                nav_df_w = nifty_nav_at_rebalance_dates(wbt, nifty_raw)
                fig_w = go.Figure()
                fig_w.add_trace(
                    go.Scatter(
                        x=wbt["date"],
                        y=wbt["portfolio_value"],
                        name=f"{label} NAV",
                        line=dict(color=color, width=2),
                    )
                )
                fig_w.add_trace(
                    go.Scatter(
                        x=nav_df_w["date"],
                        y=nav_df_w["nifty_nav"],
                        name="NIFTY buy-and-hold (aligned dates)",
                        line=dict(color="#94a3b8", width=2, dash="dot"),
                    )
                )
                add_regime_vrects(fig_w, wbt)
                fig_w.update_layout(
                    title=f"{label} — Portfolio vs NIFTY (₹100,000 start) — background = regime at rebalance",
                    xaxis_title="Date",
                    yaxis_title="NAV (₹)",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    margin=dict(t=60),
                    **_plotly_dark(),
                )
                fig_w.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig_w, width="stretch")
            except Exception as e:
                st.warning(f"Weekly equity plot failed for `{bt_path.name}`: {e}")

        w1, w2 = st.tabs(["Weekly — Ret_21d signal", "Weekly — Ret_5d signal"])
        with w1:
            _plot_weekly_equity(
                WEEKLY_BT_CSV,
                "Momentum + regime + sector (weekly Ret_21d signal)",
                "#38bdf8",
            )
        with w2:
            _plot_weekly_equity(
                WEEKLY_RET5D_BT_CSV,
                "Momentum + regime + sector (weekly Ret_5d signal)",
                "#a78bfa",
            )

    with tab_pos:
        dates = pd.to_datetime(bt["date"]).tolist()
        sel_date = st.select_slider(
            "Rebalance date",
            options=dates,
            value=dates[-1],
            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        )
        sel_ts = pd.Timestamp(sel_date).normalize()
        row = bt.loc[bt["date"].dt.normalize() == sel_ts]
        if row.empty:
            st.error("Selected date not found in backtest results.")
        else:
            r0 = row.iloc[0]
            tickers = parse_stocks_held(r0["stocks_held"])
            if not tickers:
                st.error("No `stocks_held` for this row.")
            else:
                sectors_in_picks = [get_sector(t) for t in tickers]
                sector_opts = ["All sectors"] + sorted(set(sectors_in_picks))
                sector_sel = st.selectbox("Sector filter", options=sector_opts, index=0)
                asof = pd.Timestamp(sel_date)
                asof_ns = int(asof.value)

                rows_out: list[dict[str, Any]] = []
                missing_feat: list[str] = []
                for t in tickers:
                    stem = _safe_stem_from_ticker(t)
                    fr = feature_row_at_date(stem, asof_ns, str(PROCESSED_DIR))
                    sec = get_sector(t)
                    if sector_sel != "All sectors" and sec != sector_sel:
                        continue
                    if fr is None:
                        missing_feat.append(f"{PROCESSED_DIR}/{stem}_features.parquet")
                        continue
                    ra = float(fr["Ret_21d"].iloc[0])
                    sa = fr["Sector_Alpha"].iloc[0]
                    ma = fr["Monthly_Alpha"].iloc[0]
                    alpha_show = sa if pd.notna(sa) else ma
                    rows_out.append(
                        {
                            "Ticker": t,
                            "Sector": sec,
                            "Ret_21d": ra,
                            "Alpha (Sector or Monthly)": float(alpha_show) if pd.notna(alpha_show) else np.nan,
                            "As-of feature Date": fr["Date"].iloc[0],
                        }
                    )

                if missing_feat:
                    st.error(
                        "Missing feature file(s): "
                        + ", ".join(sorted(set(missing_feat)))
                        + ". Run `features/feature_engineering.py` to build processed parquets."
                    )

                st.markdown(
                    f"**Regime:** `{r0['regime']}` · **Turnover:** {float(r0['turnover']):.4f} · "
                    f"**Friction (₹):** {float(r0['friction_cost']):.2f}"
                )

                if rows_out:
                    st.dataframe(
                        pd.DataFrame(rows_out),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Ret_21d": st.column_config.NumberColumn(format="%.4f"),
                            "Alpha (Sector or Monthly)": st.column_config.NumberColumn(format="%.4f"),
                            "As-of feature Date": st.column_config.DateColumn(),
                        },
                    )
                elif not missing_feat:
                    st.info("No rows for this sector filter.")

                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=sectors_in_picks,
                            hole=0.35,
                            marker=dict(line=dict(color="#0b1020", width=1)),
                        )
                    ]
                )
                fig_pie.update_layout(
                    title="Sector mix (this rebalance)",
                    template="plotly_dark",
                    paper_bgcolor="#0b1020",
                    showlegend=True,
                )
                st.plotly_chart(fig_pie, width="stretch")

    with tab_notes:
        tab_research_notes()


def render_model_comparison_master(
    strat: dict[str, Any] | None,
) -> None:
    """Seven-row thesis table + Sharpe / DA bar charts."""
    mj_ramt = _load_json(RAMT_DIR / "ramt_metrics.json")

    def _enrich_bt(mj: dict[str, Any] | None, bt_path: Path) -> dict[str, Any] | None:
        if mj is None and not bt_path.is_file():
            return None
        base = dict(mj or {})
        if bt_path.is_file():
            em = compute_metrics(bt_path)
            base.setdefault("cagr", em["cagr"])
            base.setdefault("max_dd", em["max_dd"])
            base.setdefault("sharpe", em["sharpe_net"])
        return base

    def cell_da(mj: dict[str, Any] | None, key: str = "directional_accuracy") -> str:
        if mj is None:
            return "N/A"
        if "DA_pct" in mj:
            return f"{float(mj['DA_pct']):.2f}%"
        if key in mj:
            return f"{100 * float(mj[key]):.2f}%"
        return "N/A"

    def cell_ic(mj: dict[str, Any] | None) -> str:
        if mj is None:
            return "N/A"
        v = mj.get("mean_IC")
        if v is None:
            return "N/A"
        return f"{float(v):.4f}"

    def cell_sharpe(mj: dict[str, Any] | None, k: str = "Sharpe") -> str:
        if mj is None:
            return "N/A"
        v = mj.get(k) or mj.get("sharpe")
        if v is None or (isinstance(v, float) and v != v):
            return "N/A"
        return f"{float(v):.2f}"

    def cell_cagr(mj: dict[str, Any] | None) -> str:
        if mj is None:
            return "N/A"
        v = mj.get("cagr")
        if v is not None and v == v:
            return f"{100 * float(v):.1f}%"
        bt_p = mj.get("_bt_path")
        if bt_p and Path(bt_p).is_file():
            m = compute_metrics(bt_p)
            return f"{100 * m['cagr']:.1f}%"
        return "N/A"

    def cell_mdd(mj: dict[str, Any] | None) -> str:
        if mj is None:
            return "N/A"
        v = mj.get("MaxDD")
        if v is not None:
            return f"{100 * float(v):.1f}%"
        v2 = mj.get("max_dd")
        if v2 is not None:
            return f"{100 * float(v2):.1f}%"
        return "N/A"

    p1x = _load_json(PHASE1_DAILY / "xgboost_metrics.json")
    p1l = _load_json(PHASE1_DAILY / "lstm_metrics.json")
    p2x = _load_json(PHASE2_MONTHLY / "xgboost_metrics.json")
    p2l = _load_json(PHASE2_MONTHLY / "lstm_metrics.json")

    p2x = _enrich_bt(p2x, PHASE2_MONTHLY / "xgboost_backtest_results.csv")
    p2l = _enrich_bt(p2l, PHASE2_MONTHLY / "lstm_backtest_results.csv")

    ramt_bt_path = RAMT_DIR / "backtest_results.csv"
    ramt_cagr_str = "N/A"
    if ramt_bt_path.is_file():
        ramt_cagr_str = f"{100 * compute_metrics(ramt_bt_path)['cagr']:.1f}%"

    rows: list[dict[str, str]] = [
        {
            "Phase": "Phase 1",
            "Model": "XGBoost",
            "Target": "Daily return",
            "DA%": cell_da(p1x),
            "Mean IC": cell_ic(p1x),
            "Sharpe": "N/A",
            "CAGR": "N/A",
            "Max DD": "N/A",
            "Notes": "Baseline; daily noise too high",
        },
        {
            "Phase": "Phase 1",
            "Model": "LSTM",
            "Target": "Daily return",
            "DA%": cell_da(p1l),
            "Mean IC": cell_ic(p1l),
            "Sharpe": "N/A",
            "CAGR": "N/A",
            "Max DD": "N/A",
            "Notes": "Underperformed XGBoost",
        },
        {
            "Phase": "Phase 2",
            "Model": "XGBoost",
            "Target": "Monthly alpha",
            "DA%": cell_da(p2x),
            "Mean IC": cell_ic(p2x),
            "Sharpe": cell_sharpe(p2x),
            "CAGR": cell_cagr(p2x),
            "Max DD": cell_mdd(p2x),
            "Notes": "Phase 2 baseline",
        },
        {
            "Phase": "Phase 2",
            "Model": "LSTM",
            "Target": "Monthly alpha",
            "DA%": cell_da(p2l),
            "Mean IC": cell_ic(p2l),
            "Sharpe": cell_sharpe(p2l),
            "CAGR": cell_cagr(p2l),
            "Max DD": cell_mdd(p2l),
            "Notes": "Same target as RAMT",
        },
        {
            "Phase": "Phase 2",
            "Model": "RAMT",
            "Target": "Monthly alpha",
            "DA%": cell_da(mj_ramt),
            "Mean IC": cell_ic(mj_ramt),
            "Sharpe": cell_sharpe(mj_ramt),
            "CAGR": ramt_cagr_str,
            "Max DD": cell_mdd(mj_ramt),
            "Notes": "Transformer + regime cross-attention",
        },
        {
            "Phase": "Diagnostic",
            "Model": "LightGBM",
            "Target": "Monthly alpha",
            "DA%": "N/A",
            "Mean IC": "~0.021 (README)",
            "Sharpe": "N/A",
            "CAGR": "N/A",
            "Max DD": "N/A",
            "Notes": "IC diagnostic from scripts/baseline_feature_ic.py — not exported to JSON",
        },
        {
            "Phase": "Final",
            "Model": "Momentum + HMM",
            "Target": "N/A",
            "DA%": "N/A",
            "Mean IC": "N/A",
            "Sharpe": f"{strat['sharpe_net']:.2f}" if strat else "N/A",
            "CAGR": f"{100 * strat['cagr']:.1f}%" if strat else "N/A",
            "Max DD": f"{100 * strat['max_dd']:.1f}%" if strat else "N/A",
            "Notes": "Rules strategy (production)",
        },
    ]

    st.subheader("Master comparison (thesis table)")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Sharpe bar (backtest-capable rows only)
    sharpe_labels: list[str] = []
    sharpe_vals: list[float] = []
    def _finite(x: Any) -> bool:
        return x is not None and (not isinstance(x, float) or x == x)

    if p2x and _finite(p2x.get("sharpe")):
        sharpe_labels.append("P2 XGBoost")
        sharpe_vals.append(float(p2x["sharpe"]))
    if p2l and _finite(p2l.get("sharpe")):
        sharpe_labels.append("P2 LSTM")
        sharpe_vals.append(float(p2l["sharpe"]))
    if mj_ramt and _finite(mj_ramt.get("Sharpe")):
        sharpe_labels.append("RAMT")
        sharpe_vals.append(float(mj_ramt["Sharpe"]))
    if strat:
        sharpe_labels.append("Momentum+HMM")
        sharpe_vals.append(float(strat["sharpe_net"]))

    if sharpe_labels:
        st.subheader("Sharpe comparison (models with backtest / strategy metrics)")
        fig_b = go.Figure(
            data=go.Bar(x=sharpe_labels, y=sharpe_vals, marker_color="#38bdf8")
        )
        fig_b.update_layout(**_plotly_dark(), yaxis_title="Sharpe", title="Sharpe")
        st.plotly_chart(fig_b, width="stretch")
    else:
        st.caption("Sharpe bar chart skipped — no Sharpe values in JSONs for Phase 2 ML rows.")

    da_labels: list[str] = []
    da_vals: list[float] = []
    for lab, mj in [
        ("P1 XGB", p1x),
        ("P1 LSTM", p1l),
        ("P2 XGB", p2x),
        ("P2 LSTM", p2l),
    ]:
        if mj and _finite(mj.get("directional_accuracy")):
            da_labels.append(lab)
            da_vals.append(float(mj["directional_accuracy"]) * 100)
        elif mj and _finite(mj.get("DA_pct")):
            da_labels.append(lab)
            da_vals.append(float(mj["DA_pct"]))
    if mj_ramt and mj_ramt.get("DA_pct") is not None:
        da_labels.append("RAMT")
        da_vals.append(float(mj_ramt["DA_pct"]))

    if da_labels:
        st.subheader("Directional accuracy (%)")
        fig_d = go.Figure(data=go.Bar(x=da_labels, y=da_vals, marker_color="#a78bfa"))
        fig_d.update_layout(**_plotly_dark(), yaxis_title="DA %", title="Directional accuracy")
        st.plotly_chart(fig_d, width="stretch")


def tab_research_notes() -> None:
    st.markdown(
        """
## Project pivot

We initially trained a **regime-adaptive multimodal transformer (RAMT)** on cross-sectional
NIFTY 200 features. The model did not produce a stable positive information coefficient (IC)
on held-out months, so we **stopped using it as a production signal**.

The current, ground-truth research track is a **transparent rules-based strategy**:
cross-sectional **21-day momentum (`Ret_21d`)**, **HMM regime** position sizing
(BULL / HIGH_VOL / BEAR), and **one name per sector** for diversification.

---

## Diagnostic (IC)

On comparable setups, a **LightGBM** baseline showed **IC ≈ +0.021** on ranked targets,
while **RAMT was roughly −0.02 IC** — consistent with the pivot away from the transformer
as the primary alpha source.

---

## Limitations

- **Static universe** (NIFTY 200–style list) — survivorship and membership drift are not modeled.
- **~2-year test window** (2024–2026) — one macro regime; results are illustrative, not a guarantee.
- **Indian equities** — liquidity, taxes, and execution differ from paper backtests.

---

## Future work

- **Volatility filter** on entries / sizing.
- **Ensemble** with gradient boosting (e.g. LightGBM) where IC is positive.
- **Multi-cycle validation** (longer history, walk-forward segments).
"""
    )


def main() -> None:
    st.title("NIFTY 200 research — model comparison")
    st.caption(
        "Pick a model or section in the sidebar. RAMT, LSTM, and XGBoost (Phase 1/2) plus "
        "the production momentum + HMM strategy."
    )

    missing_nifty = not NIFTY_PARQUET.is_file()
    missing_bt = not BACKTEST_CSV.is_file()

    _SECTIONS = [
        "RAMT transformer",
        "Production strategy (momentum + HMM)",
        "LSTM",
        "XGBoost",
        "Model comparison",
    ]

    with st.sidebar:
        st.subheader("Results")
        section = st.radio(
            "Choose view",
            options=_SECTIONS,
            index=0,
            help="Switch between model outputs and the cross-model comparison table.",
        )
        st.divider()
        st.subheader("Data sources")
        st.text(f"Production backtest: {BACKTEST_CSV}")
        st.text(f"RAMT: {RAMT_DIR}")
        st.text(f"Phase 1 baselines (XGB/LSTM daily): {BASELINE_WALKFORWARD}")
        st.text(f"Phase 2 monthly: {PHASE2_MONTHLY}")
        st.text(f"NIFTY raw: {NIFTY_PARQUET}")
        if not missing_bt:
            mtime = pd.Timestamp.fromtimestamp(BACKTEST_CSV.stat().st_mtime)
            st.caption(f"Production backtest mtime: {mtime.strftime('%Y-%m-%d %H:%M')}")

    bt = None
    nifty_raw = None
    strat = None
    bench = None
    if not missing_nifty:
        try:
            nifty_raw = load_nifty_prices(str(NIFTY_PARQUET))
        except Exception as e:
            st.sidebar.warning(f"NIFTY load: {e}")
    if not missing_bt and nifty_raw is not None:
        try:
            bt = load_backtest_csv(str(BACKTEST_CSV))
            strat = compute_metrics(BACKTEST_CSV)
            bench = compute_nifty_benchmark(
                NIFTY_PARQUET, bt["date"].iloc[0], bt["date"].iloc[-1]
            )
        except Exception as e:
            st.sidebar.warning(f"Production backtest load: {e}")

    if section == "RAMT transformer":
        render_ramt_transformer_section()

    elif section == "Production strategy (momentum + HMM)":
        st.subheader("Production strategy — Momentum + regime + sector")
        st.caption("Rules-based portfolio from `results/final_strategy/backtest_results.csv`.")
        if missing_nifty:
            st.error(f"Missing `{NIFTY_PARQUET}`.")
        elif missing_bt or bt is None or strat is None or bench is None:
            st.warning(f"This section needs `{BACKTEST_CSV}` and a valid NIFTY series.")
        else:
            render_momentum_strategy_tabs(bt, nifty_raw, strat, bench)

    elif section == "LSTM":
        st.caption("LSTM — artifacts under `results/phase1_daily/` and `results/phase2_monthly/`.")
        lp1, lp2 = st.tabs(["Phase 1 (daily returns)", "Phase 2 (monthly alpha)"])
        with lp1:
            render_phase1_daily_block(
                "LSTM",
                PHASE1_DAILY / "lstm_predictions.csv",
                PHASE1_DAILY / "lstm_metrics.json",
            )
        with lp2:
            render_phase2_monthly_block(
                "LSTM",
                PHASE2_MONTHLY / "lstm_predictions.csv",
                PHASE2_MONTHLY / "lstm_metrics.json",
                PHASE2_MONTHLY / "lstm_backtest_results.csv",
                baseline_callout=False,
            )

    elif section == "XGBoost":
        st.caption("XGBoost — same folder layout as LSTM.")
        xp1, xp2 = st.tabs(["Phase 1 (daily returns)", "Phase 2 (monthly alpha)"])
        with xp1:
            render_phase1_daily_block(
                "XGBoost",
                PHASE1_DAILY / "xgboost_predictions.csv",
                PHASE1_DAILY / "xgboost_metrics.json",
            )
        with xp2:
            render_phase2_monthly_block(
                "XGBoost",
                PHASE2_MONTHLY / "xgboost_predictions.csv",
                PHASE2_MONTHLY / "xgboost_metrics.json",
                PHASE2_MONTHLY / "xgboost_backtest_results.csv",
                baseline_callout=True,
            )

    else:
        render_model_comparison_master(strat)


if __name__ == "__main__":
    main()
