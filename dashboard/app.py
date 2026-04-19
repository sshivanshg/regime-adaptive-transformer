"""
Momentum + Regime + Sector — NIFTY 200 research dashboard.

Loads all strategy metrics from ``results/backtest_results.csv`` and benchmarks
from ``data/raw/_NSEI.parquet``. No live model inference.
"""

from __future__ import annotations

import ast
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

BACKTEST_CSV = ROOT / "results" / "backtest_results.csv"
NIFTY_PARQUET = ROOT / "data" / "raw" / "_NSEI.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"

# Optional archived comparison CSVs (if present)
ARCHIVE_RAMT_BACKTEST = ROOT / "results" / "archive" / "ramt_backtest_results.csv"
ARCHIVE_MOM_NO_SECTOR = ROOT / "results" / "archive" / "momentum_regime_no_sector_backtest.csv"

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


def compute_metrics(bt_path: str | Path, capital: float = 100_000) -> dict[str, Any]:
    bt = pd.read_csv(bt_path, parse_dates=["date"])
    r = bt["portfolio_return"].dropna()
    nav = bt["portfolio_value"].values.astype(float)
    start_ts = bt["date"].iloc[0]
    end_ts = bt["date"].iloc[-1]
    span_years = (end_ts - start_ts).days / 365.25

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
    st.title("Momentum + Regime · NIFTY 200")
    st.caption(
        "Rules-based momentum, HMM regime sizing, and sector caps — benchmarked vs NIFTY "
        "(buy-and-hold, adjusted close)."
    )

    missing: list[str] = []
    if not BACKTEST_CSV.is_file():
        missing.append(f"`{BACKTEST_CSV}` — run the backtest pipeline (e.g. `models/run_final_2024_2026.py`).")
    if not NIFTY_PARQUET.is_file():
        missing.append(f"`{NIFTY_PARQUET}` — ensure raw benchmark data is present.")

    with st.sidebar:
        st.subheader("Data sources")
        st.text(f"Backtest: {BACKTEST_CSV}")
        st.text(f"NIFTY raw: {NIFTY_PARQUET}")
        st.text(f"Features: {PROCESSED_DIR}/")
        if BACKTEST_CSV.is_file():
            mtime = pd.Timestamp.fromtimestamp(BACKTEST_CSV.stat().st_mtime)
            st.caption(f"Backtest file mtime: {mtime.strftime('%Y-%m-%d %H:%M')}")
        if not missing and BACKTEST_CSV.is_file():
            bt1 = pd.read_csv(BACKTEST_CSV, parse_dates=["date"])
            st.success(
                f"As-of window: **{bt1['date'].iloc[0].date()}** → **{bt1['date'].iloc[-1].date()}** "
                f"({len(bt1)} rebalance rows)."
            )

    tab_perf, tab_bench, tab_pos, tab_notes = st.tabs(
        ["Strategy Performance", "Strategy vs Benchmarks", "Positions", "Research notes"]
    )

    if missing:
        for m in missing:
            st.error(m)
        return

    try:
        bt = load_backtest_csv(str(BACKTEST_CSV))
        nifty_raw = load_nifty_prices(str(NIFTY_PARQUET))
        strat = compute_metrics(BACKTEST_CSV)
        bench = compute_nifty_benchmark(
            NIFTY_PARQUET, bt["date"].iloc[0], bt["date"].iloc[-1]
        )
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        return
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    nifty_win = nifty_inter_rebalance_win_rate(bt, nifty_raw)

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
            "CSV exists under `results/archive/`."
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

                # Pie: sectors for all 5 picks (ignore filter for diversity view)
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


if __name__ == "__main__":
    main()
