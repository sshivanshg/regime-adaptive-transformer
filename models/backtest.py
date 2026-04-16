"""
Portfolio Backtest 2015-2024

Monthly rebalancing strategy:
1. First trading day of month
2. Check HMM regime
3. If BULL: top N RAMT-ranked stocks at 100% allocation
4. If HIGH_VOL: top 3 at 50% allocation
5. If BEAR: top 5 at 20% allocation
6. Hold until next month
7. Repeat

Metrics to report:
- Annual return vs NIFTY
- Sharpe ratio
- Max drawdown
- Win rate (% months beating NIFTY)
- Best/worst month
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REAL_2026_REBALANCE_FRICTION_RATE = 0.0022
LEGACY_REBALANCE_FRICTION_RATE = 0.0020


# -----------------------------------------------------------------------------
# Kelly criterion & signal-weighted baskets
# -----------------------------------------------------------------------------


def kelly_optimal_fraction(p: float, b: float) -> float:
    """
    Full Kelly fraction for binary outcomes: f* = (p*b - q) / b, q = 1 - p.

    ``p``: win probability (e.g. directional accuracy).
    ``b``: win/loss ratio (avg win / |avg loss| on losing trades).
    """
    q = 1.0 - float(p)
    b = float(b)
    if b <= 1e-12:
        return 0.0
    return float((p * b - q) / b)


def estimate_win_loss_ratio(actual_alpha: pd.Series | np.ndarray) -> float:
    """Empirical b = mean(win) / |mean(loss)| from realized alphas."""
    s = pd.Series(np.asarray(actual_alpha, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 1.0
    wins = s[s > 0]
    losses = s[s < 0]
    if len(wins) < 1 or len(losses) < 1:
        return 1.0
    aw = float(wins.mean())
    al = float(abs(losses.mean()))
    return float(aw / max(al, 1e-12))


def kelly_b_from_predicted_alpha_margin(
    pred_alpha: np.ndarray,
    *,
    scale: float = 0.02,
) -> float:
    """
    Map predicted-alpha spread (top vs bottom of the selected basket) to Kelly odds ``b``.

    ``scale`` is a reference alpha margin (e.g. 0.02 = 2 percentage points) so that
    wider predicted separation implies larger ``b`` and a higher Kelly fraction.
    """
    x = np.asarray(pred_alpha, dtype=float)
    if x.size == 0:
        return 1.0
    margin = float(np.max(x) - np.min(x))
    return max(1.0, 1.0 + margin / max(float(scale), 1e-12))


def basket_weights_from_alpha(
    predicted_alpha: np.ndarray,
    *,
    temperature: float = 1.0,
    blend_equal: float = 0.0,
) -> np.ndarray:
    """
    Softmax weights over ``predicted_alpha`` so higher-alpha names get more capital.
    ``blend_equal`` in [0, 1] mixes toward equal weight for stability.
    """
    x = np.asarray(predicted_alpha, dtype=float)
    if x.size == 0:
        return np.array([])
    x = x - np.max(x)
    e = np.exp(x / max(float(temperature), 1e-6))
    w = e / np.sum(e)
    if blend_equal > 0.0:
        n = len(w)
        eq = np.full(n, 1.0 / n)
        w = (1.0 - blend_equal) * w + blend_equal * eq
        w = w / np.sum(w)
    return w


def _nav_fractions_kelly(
    predicted_alpha: np.ndarray,
    position_size: float,
    max_weight: float,
    *,
    temperature: float,
    blend_equal: float,
) -> tuple[np.ndarray, float]:
    """
    NAV fractions per name (sum = invested sleeve before per-name cap), then apply ``max_weight``.
    Returns (alloc_fractions, invested_after_cap).
    """
    w = basket_weights_from_alpha(
        predicted_alpha, temperature=temperature, blend_equal=blend_equal
    )
    alloc = position_size * w
    alloc = np.minimum(alloc, max_weight)
    invested = float(np.sum(alloc))
    return alloc, invested


def _load_nifty_benchmark_raw(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load NIFTY benchmark OHLCV from Parquet (preferred) or legacy `_NSEI_raw.csv`.

    Ensures `Adj Close` exists for downstream feature helpers.
    """
    rdir = Path(raw_dir)
    pq = rdir / "_NSEI.parquet"
    csv = rdir / "_NSEI_raw.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
        df["Date"] = pd.to_datetime(df["Date"])
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"].astype(float)
        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                df[col] = df["Adj Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 1.0
    else:
        raise FileNotFoundError(
            f"NIFTY benchmark not found: expected {pq} or {csv}"
        )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def ensure_nifty_features_parquet(processed_dir: str | Path, raw_dir: str | Path) -> str:
    """
    Return path to ``_NSEI_features.parquet``, building it if missing.

    Uses the same pipeline as ``features.feature_engineering`` so the index file has
    the full feature column set (including ``HMM_Regime``) as stock files.
    """
    pdir = Path(processed_dir)
    rdir = Path(raw_dir)
    out = pdir / "_NSEI_features.parquet"
    if out.exists():
        return str(out)

    pdir.mkdir(parents=True, exist_ok=True)

    from features.feature_engineering import (
        _download_benchmark_if_missing,
        _read_raw_equity,
        load_macro_series,
        process_raw_equity_path,
    )

    bench_path = _download_benchmark_if_missing()
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Cannot build NIFTY features: missing benchmark Parquet/CSV under {rdir}"
        )

    nifty_df = _read_raw_equity(bench_path)
    macro_data = load_macro_series(rdir)
    _, written = process_raw_equity_path(bench_path, nifty_df, macro_data, pdir)
    if written is None or not written.exists():
        raise RuntimeError("Failed to materialize _NSEI_features.parquet from raw NIFTY data.")
    return str(written.resolve())


def resolve_nifty_features_path(nifty_features_path: str, raw_dir: str) -> str:
    """Use existing processed Parquet if present; otherwise build from raw NIFTY."""
    p = Path(nifty_features_path)
    if p.exists():
        return str(p.resolve())
    processed_dir = Path(raw_dir).resolve().parent / "processed"
    return ensure_nifty_features_parquet(processed_dir, raw_dir)


def _load_price_series(raw_path: str) -> pd.Series:
    p = pd.read_parquet(raw_path)
    p["Date"] = pd.to_datetime(p["Date"])
    p = p.sort_values("Date")
    # Use Adj Close for integrity under splits/bonuses
    s = p.set_index("Date")["Adj Close"].astype(float)
    return s


def _period_return_from_prices(price_start: float, price_end: float) -> float:
    if not np.isfinite(price_start) or not np.isfinite(price_end) or abs(price_start) <= 1e-12:
        return 0.0
    period_return = (price_end - price_start) / price_start
    return float(period_return)


def _window_period_return(window: pd.Series) -> float:
    if window.empty:
        return 0.0
    price_start = float(window.iloc[0])
    price_end = float(window.iloc[-1])
    return _period_return_from_prices(price_start, price_end)


def build_rebalance_regime_df(
    nifty_features_path: str,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build regime series aligned to explicit rebalance dates.
    If exact date is missing in features, use last available previous value.
    """
    f = pd.read_parquet(nifty_features_path)
    f["Date"] = pd.to_datetime(f["Date"])
    f = f.sort_values("Date").set_index("Date")["HMM_Regime"].astype(float).ffill()
    out = []
    for d in rebalance_dates:
        # take last available on/before date
        sel = f.loc[:d]
        if sel.empty:
            continue
        out.append({"Date": pd.Timestamp(d), "regime": int(sel.iloc[-1])})
    return pd.DataFrame(out)


def build_monthly_regime_df(
    nifty_features_path: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Build a month-start regime series from processed NIFTY features.

    Returns DataFrame with columns: Date, regime
    where Date is month start (first trading day present in that month).
    """
    df = pd.read_parquet(nifty_features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    if df.empty or "HMM_Regime" not in df.columns:
        return pd.DataFrame(columns=["Date", "regime"])

    df["Month"] = df["Date"].dt.to_period("M")
    first_rows = df.groupby("Month", as_index=False).head(1)
    out = first_rows[["Date", "HMM_Regime"]].rename(columns={"HMM_Regime": "regime"})
    out["regime"] = out["regime"].astype(int)
    return out.reset_index(drop=True)


def compute_nifty_monthly_returns(
    nifty_raw_path: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Approximate month-ahead NIFTY return as 21-trading-day forward log return.

    Returns DataFrame: date, nifty_return
    where date is month start (first trading day present in that month).
    """
    df = pd.read_parquet(nifty_raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    if df.empty:
        return pd.DataFrame(columns=["date", "nifty_return"])

    px = df["Adj Close"].astype(float).replace(0.0, np.nan)
    r1 = px / px.shift(1)
    lr = np.log(r1.where(r1 > 0.0))
    df["fwd_21"] = lr.rolling(21).sum().shift(-21)
    df["Month"] = df["Date"].dt.to_period("M")
    ms = df.groupby("Month", as_index=False).head(1)
    out = ms[["Date", "fwd_21"]].rename(columns={"Date": "date", "fwd_21": "nifty_return"})
    out["nifty_return"] = out["nifty_return"].fillna(0.0).astype(float)
    return out.reset_index(drop=True)


def run_backtest(
    predictions_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    start: str = "2016-01-01",
    end: str = "2024-12-31",
    top_n: int = 5,
    capital: float = 50000,
    raw_dir: str | None = None,
    *,
    use_kelly_weights: bool = True,
    kelly_p: float = 0.5238,
    kelly_b: float | None = None,
    kelly_temperature: float = 1.0,
    kelly_blend_equal: float = 0.0,
    kelly_fractional: float = 0.25,
    kelly_scale_position: bool = False,
) -> pd.DataFrame:
    """
    Full portfolio backtest.

    **Geometric growth:** Each month’s ``portfolio_return`` is applied to end-of-month NAV
    (``portfolio_value``), so profits reinvest before the next rebalance — not a fixed
    notional each period.

    **Kelly-style sizing (optional):** When ``use_kelly_weights`` is True, intra-month
    weights follow a softmax over ``predicted_alpha`` (concentrate on top signals). The
    classical Kelly fraction ``f* = (p*b - q)/b`` is computed from ``kelly_p`` and
    ``kelly_b`` (or empirical ``b`` from ``actual_alpha`` when ``kelly_b`` is None).
    Use ``kelly_scale_position`` to optionally scale the regime ``position_size`` by a
    fractional-Kelly factor (conservative).

    predictions_df columns:
      Date, Ticker, predicted_alpha, actual_alpha
      Optional: price_start, price_end

    If ``price_start``/``price_end`` are present, or ``raw_dir`` is supplied so prices
    can be loaded from disk, realized period returns use total return rather than
    ``actual_alpha``.

    regime_df columns:
      Date, regime (0/1/2)

    Returns:
      One row per traded month with ``portfolio_return``, ``portfolio_value`` (compounded NAV),
      and ``cumulative_return`` vs initial ``capital``.
    """
    results: list[dict[str, object]] = []
    monthly_dates = pd.date_range(start, end, freq="MS")

    predictions_df = predictions_df.copy()
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
    regime_df = regime_df.copy()
    regime_df["Date"] = pd.to_datetime(regime_df["Date"])

    kb = float(kelly_b) if kelly_b is not None else estimate_win_loss_ratio(predictions_df["actual_alpha"])
    kelly_f_star_global = kelly_optimal_fraction(float(kelly_p), kb)

    pv = float(capital)
    price_cache: dict[str, pd.Series] = {}
    raw_root = Path(raw_dir) if raw_dir is not None else None

    def get_prices(ticker: str) -> pd.Series:
        if raw_root is None:
            raise ValueError("raw_dir is required to compute realized total returns from price history.")
        if ticker in price_cache:
            return price_cache[ticker]
        path = raw_root / f"{ticker}.parquet"
        price_cache[ticker] = _load_price_series(str(path))
        return price_cache[ticker]

    for i, date in enumerate(monthly_dates[:-1]):
        _next_date = monthly_dates[i + 1]

        month_regime = regime_df[regime_df["Date"] == date]["regime"].values
        if len(month_regime) == 0:
            continue
        regime = int(month_regime[0])

        if regime == 2:  # Bear — fractional toe-hold
            position_size = 0.2
            top_n_regime = min(5, top_n)
        elif regime == 0:  # High vol
            position_size = 0.5
            top_n_regime = 3
        else:  # Bull
            position_size = 1.0
            top_n_regime = top_n

        month_preds = predictions_df[predictions_df["Date"] == date].nlargest(
            top_n_regime, "predicted_alpha"
        )
        if month_preds.empty:
            continue

        if {"price_start", "price_end"}.issubset(month_preds.columns):
            price_start = month_preds["price_start"].values.astype(float)
            price_end = month_preds["price_end"].values.astype(float)
            actual_returns = np.asarray(
                [
                    _period_return_from_prices(start_px, end_px)
                    for start_px, end_px in zip(price_start, price_end)
                ],
                dtype=float,
            )
        elif raw_root is not None:
            actual_return_list = []
            for ticker in month_preds["Ticker"]:
                prices = get_prices(ticker)
                window = prices.loc[(prices.index >= date) & (prices.index < _next_date)]
                actual_return_list.append(_window_period_return(window))
            actual_returns = np.asarray(actual_return_list, dtype=float)
        else:
            actual_returns = month_preds["actual_alpha"].values.astype(float)
        pred_alpha = month_preds["predicted_alpha"].values.astype(float)
        ps = float(position_size)
        if kelly_scale_position and kelly_f_star_global > 0.0:
            ps = float(
                ps
                * min(
                    1.0,
                    max(0.2, kelly_fractional * kelly_f_star_global / 0.05),
                )
            )
        if use_kelly_weights:
            w = basket_weights_from_alpha(
                pred_alpha,
                temperature=kelly_temperature,
                blend_equal=kelly_blend_equal,
            )
            basket_ret = float(np.dot(w, actual_returns))
            w_max = float(np.max(w))
        else:
            basket_ret = float(np.mean(actual_returns))
            w_max = float("nan")
        portfolio_return = basket_ret * ps

        pv_start = pv
        pv = pv * (1.0 + portfolio_return)

        results.append(
            {
                "date": date,
                "portfolio_return": portfolio_return,
                "portfolio_value_start": pv_start,
                "portfolio_value": pv,
                "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                "stocks_held": month_preds["Ticker"].tolist(),
                "cash": False,
                "kelly_f_star": kelly_f_star_global,
                "kelly_b_used": kb,
                "kelly_largest_weight": w_max,
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    results_df["cumulative_return"] = results_df["portfolio_value"] / float(capital) - 1.0

    monthly_returns = results_df["portfolio_return"].values.astype(float)
    n_m = len(monthly_returns)
    total_ret = float(results_df["portfolio_value"].iloc[-1] / float(capital) - 1.0)
    years = n_m / 12.0
    cagr_pct = ((1.0 + total_ret) ** (1.0 / years) - 1.0) * 100.0 if years > 1e-9 else 0.0
    sharpe = (np.mean(monthly_returns) / (np.std(monthly_returns) + 1e-8)) * np.sqrt(12)
    max_dd = compute_max_drawdown(monthly_returns)
    win_rate = float(np.mean(monthly_returns > 0) * 100)

    print(f"\n{'='*50}")
    print("BACKTEST RESULTS 2016-2024")
    print(f"{'='*50}")
    print(f"CAGR:           {cagr_pct:.1f}%  (reinvested NAV each month)")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print(f"Max Drawdown:   {max_dd*100:.1f}%")
    print(f"Win Rate:       {win_rate:.1f}% months")
    print(
        f"Capital: ₹{capital:,} → "
        f"₹{float(results_df['portfolio_value'].iloc[-1]):,.0f} "
        f"(+{total_ret*100:.1f}% total)"
    )

    return results_df


def run_backtest_daily(
    predictions_df: pd.DataFrame,
    nifty_features_path: str,
    raw_dir: str,
    start: str,
    end: str,
    step_size: int = 21,
    top_n: int = 5,
    capital: float = 100000,
    stop_loss: float = 0.07,
    stop_loss_bear: float = 0.05,
    max_weight: float = 0.20,
    portfolio_dd_cash_trigger: float = 0.15,
    rebalance_friction_rate: float = REAL_2026_REBALANCE_FRICTION_RATE,
    turnover_penalty_score: float = 0.0,
    *,
    use_kelly_weights: bool = True,
    kelly_p: float = 0.5238,
    kelly_b: float | None = None,
    kelly_use_predicted_margin: bool = False,
    kelly_alpha_margin_scale: float = 0.02,
    kelly_temperature: float = 1.0,
    kelly_blend_equal: float = 0.0,
    kelly_fractional: float = 0.25,
    kelly_scale_position: bool = False,
) -> pd.DataFrame:
    """
    Daily-price backtest with risk rules.

    **Geometric growth:** Each window starts from the prior window’s ending NAV:
    ``portfolio_value_start`` equals the previous row’s ``portfolio_value`` (rolling
    ``pv``), then ``pv_{t+1} = pv_t * (1 + port_ret)``. Friction applies to turnover at
    that rebalance so compounding reflects post-cost equity.

    **Kelly-style baskets:** When ``use_kelly_weights`` is True, capital is split across
    names using a softmax over ``predicted_alpha`` (see ``run_backtest``). Otherwise
    the legacy equal-weight sleeve applies. When ``kelly_use_predicted_margin`` is True,
    Kelly odds ``b`` come from the predicted-alpha margin each rebalance; otherwise
    ``kelly_b`` or the empirical win/loss ratio is used.

    - Rebalance every `step_size` trading days on NIFTY calendar.
    - **Friction:** each rebalance deducts ``rebalance_friction_rate`` (default **0.22%**)
      from **total trade value** = equity notional at rebalance, ``pv_start * invested``
      (STT + slippage combined; applied every window we deploy capital).
    - Stop-loss per stock (intraperiod): ``stop_loss`` by default; in BEAR use ``stop_loss_bear`` (5%).
    - Max weight per stock: cap at `max_weight`, remainder stays cash.
    - If portfolio return <= -portfolio_dd_cash_trigger in a window:
        force next window to cash.

    Returns one row per rebalance window with trade/hold details.
    """
    preds = predictions_df.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    effective_friction_rate = float(rebalance_friction_rate)
    if np.isclose(effective_friction_rate, LEGACY_REBALANCE_FRICTION_RATE):
        effective_friction_rate = REAL_2026_REBALANCE_FRICTION_RATE

    kb_global = float(kelly_b) if kelly_b is not None else estimate_win_loss_ratio(preds["actual_alpha"])
    kelly_f_star_global = kelly_optimal_fraction(float(kelly_p), kb_global)

    nifty_features_path = resolve_nifty_features_path(nifty_features_path, raw_dir)

    nifty_raw = _load_nifty_benchmark_raw(raw_dir)
    cal = pd.DatetimeIndex(nifty_raw["Date"])
    cal = cal[(cal >= start_ts) & (cal <= end_ts)]
    rebal = cal[::step_size]
    if len(rebal) < 2:
        return pd.DataFrame()

    regime_df = build_rebalance_regime_df(nifty_features_path, rebal)
    regime_df = regime_df.set_index("Date")["regime"]

    # Preload price series lazily
    price_cache: dict[str, pd.Series] = {}

    def get_prices(ticker: str) -> pd.Series:
        if ticker in price_cache:
            return price_cache[ticker]
        path = f"{raw_dir}/{ticker}.parquet"
        price_cache[ticker] = _load_price_series(path)
        return price_cache[ticker]

    forced_cash_next = False
    results: list[dict[str, object]] = []
    pv = float(capital)
    prev_holdings: set[str] = set()

    for i in range(len(rebal) - 1):
        d0 = pd.Timestamp(rebal[i])
        d1 = pd.Timestamp(rebal[i + 1])

        regime = int(regime_df.loc[:d0].iloc[-1]) if not regime_df.loc[:d0].empty else 1
        pv_start = float(pv)
        if forced_cash_next:
            results.append(
                {
                    "date": d0,
                    "portfolio_return": 0.0,
                    "regime": "RISK_OFF",
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value_start": pv_start,
                    "portfolio_value": pv,
                }
            )
            forced_cash_next = False
            prev_holdings = set()
            continue

        if regime == 2:
            position_size = 0.2
            n_sel = min(5, top_n)
            sl_stock = stop_loss_bear
        elif regime == 0:
            position_size = 0.5
            n_sel = 3
            sl_stock = stop_loss
        else:
            position_size = 1.0
            n_sel = top_n
            sl_stock = stop_loss

        month_df = preds[preds["Date"] == d0].copy()
        if month_df.empty:
            results.append(
                {
                    "date": d0,
                    "portfolio_return": 0.0,
                    "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value_start": pv_start,
                    "portfolio_value": pv,
                }
            )
            prev_holdings = set()
            continue

        # Optional turnover-aware selection: penalize NEW names slightly so we don't churn
        if turnover_penalty_score > 0 and prev_holdings:
            month_df["score_adj"] = month_df["predicted_alpha"]
            month_df.loc[~month_df["Ticker"].isin(prev_holdings), "score_adj"] = (
                month_df.loc[~month_df["Ticker"].isin(prev_holdings), "score_adj"] - turnover_penalty_score
            )
            month_preds = month_df.nlargest(n_sel, "score_adj")
        else:
            month_preds = month_df.nlargest(n_sel, "predicted_alpha")
        if month_preds.empty:
            results.append(
                {
                    "date": d0,
                    "portfolio_return": 0.0,
                    "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                    "stocks_held": [],
                    "cash": True,
                    "portfolio_value_start": pv_start,
                    "portfolio_value": pv,
                }
            )
            continue

        tickers = month_preds["Ticker"].tolist()
        pred_a = month_preds["predicted_alpha"].values.astype(float)
        ps = float(position_size)
        if kelly_use_predicted_margin:
            kb_here = kelly_b_from_predicted_alpha_margin(
                pred_a, scale=float(kelly_alpha_margin_scale)
            )
            f_star = kelly_optimal_fraction(float(kelly_p), kb_here)
        else:
            kb_here = kb_global
            f_star = kelly_f_star_global
        if kelly_scale_position and f_star > 0.0:
            ps = float(
                ps
                * min(
                    1.0,
                    max(0.2, kelly_fractional * f_star / 0.05),
                )
            )
        n_names = len(tickers)
        if use_kelly_weights:
            alloc, invested = _nav_fractions_kelly(
                pred_a,
                ps,
                max_weight,
                temperature=kelly_temperature,
                blend_equal=kelly_blend_equal,
            )
        else:
            w_each = min(1.0 / n_names, max_weight)
            alloc = np.full(n_names, w_each * ps, dtype=float)
            invested = float(np.sum(alloc))
        cash_weight = 1.0 - invested

        # Turnover (diagnostic): fraction of sleeve that changed names vs prior rebalance
        new_holdings = set(tickers)
        if not prev_holdings:
            turnover = invested  # entering positions from cash
        else:
            changed = len(new_holdings.symmetric_difference(prev_holdings))
            denom = max(1, len(new_holdings.union(prev_holdings)))
            turnover = invested * (changed / denom)

        # 2026 fee schedule: 0.1% buy + 0.1% sell + GST/levies on total trade value.
        total_trade_value = float(pv_start * invested)
        friction_fee = total_trade_value * effective_friction_rate

        stock_rets = []
        stopped = []
        for t in tickers:
            px = get_prices(t)
            window = px.loc[(px.index >= d0) & (px.index < d1)]
            if window.empty:
                stock_rets.append(0.0)
                continue
            entry = float(window.iloc[0])
            # Audit-only stop breach flag; realized return still reflects the full window.
            min_px = float(window.min())
            if (min_px / entry - 1.0) <= -sl_stock:
                stopped.append(t)
            stock_rets.append(_window_period_return(window))

        stock_rets_arr = np.asarray(stock_rets, dtype=float)
        if use_kelly_weights:
            port_ret_gross = float(np.dot(alloc, stock_rets_arr)) if len(alloc) else 0.0
        else:
            gross_stock_ret = float(np.mean(stock_rets_arr)) if len(stock_rets_arr) else 0.0
            port_ret_gross = invested * gross_stock_ret  # cash returns 0
        port_ret = port_ret_gross - (friction_fee / pv_start if pv_start > 0 else 0.0)

        if port_ret <= -portfolio_dd_cash_trigger:
            forced_cash_next = True

        pv = pv_start * (1.0 + port_ret)
        results.append(
            {
                "date": d0,
                "portfolio_return": port_ret,
                "portfolio_return_gross": port_ret_gross,
                "trade_value": total_trade_value,
                "friction_cost": float(friction_fee),
                "rebalance_friction_rate": float(effective_friction_rate),
                "turnover": float(turnover),
                "regime": ["HIGH_VOL", "BULL", "BEAR"][regime],
                "stocks_held": tickers,
                "stops_hit": stopped,
                "cash": False,
                "portfolio_value_start": pv_start,
                "portfolio_value": pv,
                "cash_weight": cash_weight,
                "invested_weight": invested,
                "kelly_f_star": float(f_star),
                "kelly_b_used": float(kb_here),
                "kelly_largest_weight": float(np.max(alloc)) if len(alloc) else 0.0,
            }
        )
        prev_holdings = new_holdings

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["cumulative_return"] = (df["portfolio_value"] / float(capital)) - 1.0
    return df


def compute_max_drawdown(returns) -> float:
    cumulative = np.cumprod(1 + np.array(returns, dtype=float))
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


if __name__ == "__main__":
    print("Loading predictions...")
    # Load from results/ranking_predictions.csv
    # Run backtest
    # Print results
    raise SystemExit(
        "Backtest module scaffolded. "
        "Not runnable until ranking predictions + regime series are produced."
    )
