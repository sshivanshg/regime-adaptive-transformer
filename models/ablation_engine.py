from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.backtest import run_backtest_daily


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_momentum: bool = True
    use_hmm_regime: bool = True
    use_finbert_sentiment: bool = False
    use_lora_adapters: bool = False
    use_chronos_lora: bool = False
    top_n: int = 5
    capital: float = 100000.0


def _normalize_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(".", "_", regex=False)


def _load_momentum_panel(processed_dir: Path, start: str, end: str) -> pd.DataFrame:
    files = sorted(
        p
        for p in processed_dir.glob("*_features.parquet")
        if p.name != "_NSEI_features.parquet"
    )
    if not files:
        raise FileNotFoundError(f"No processed files in {processed_dir}")

    chunks: list[pd.DataFrame] = []
    for p in files:
        d = pd.read_parquet(p, columns=["Date", "Ticker", "Ret_21d", "Sector_Alpha", "Monthly_Alpha"])
        chunks.append(d)

    panel = pd.concat(chunks, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.tz_localize(None)
    panel["Ticker"] = _normalize_ticker(panel["Ticker"])

    target = "Sector_Alpha" if panel["Sector_Alpha"].notna().any() else "Monthly_Alpha"
    panel = panel.rename(columns={"Ret_21d": "momentum", target: "actual_alpha"})

    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    panel = panel[(panel["Date"] >= s) & (panel["Date"] <= e)].copy()
    panel["momentum"] = pd.to_numeric(panel["momentum"], errors="coerce")
    panel["actual_alpha"] = pd.to_numeric(panel["actual_alpha"], errors="coerce").fillna(0.0)
    panel = panel.dropna(subset=["momentum"])

    # Data leakage control: use T signal for trade at T+1, implemented as per-ticker lag.
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    panel["momentum_t1"] = panel.groupby("Ticker")["momentum"].shift(1)
    panel = panel.dropna(subset=["momentum_t1"]).copy()
    return panel


def _load_sentiment(sentiment_path: Path) -> pd.DataFrame:
    s = pd.read_parquet(sentiment_path)
    needed = {"Date", "Ticker", "sentiment_score", "sentiment_confidence"}
    miss = needed.difference(s.columns)
    if miss:
        raise ValueError(f"Sentiment file missing columns: {sorted(miss)}")

    s = s.copy()
    s["Date"] = pd.to_datetime(s["Date"]).dt.tz_localize(None)
    s["Ticker"] = _normalize_ticker(s["Ticker"])
    s["sentiment_score"] = pd.to_numeric(s["sentiment_score"], errors="coerce").fillna(0.0)
    s["sentiment_confidence"] = pd.to_numeric(s["sentiment_confidence"], errors="coerce").fillna(0.0)
    s = s.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    s["sentiment_t1"] = s.groupby("Ticker")["sentiment_score"].shift(1)
    s["sentiment_conf_t1"] = s.groupby("Ticker")["sentiment_confidence"].shift(1)
    return s.dropna(subset=["sentiment_t1"])


def _build_predictions(
    panel: pd.DataFrame,
    cfg: AblationConfig,
    sentiment_df: pd.DataFrame | None,
) -> pd.DataFrame:
    x = panel.copy()

    if cfg.use_finbert_sentiment:
        if sentiment_df is None:
            raise ValueError("Sentiment is enabled but no sentiment dataframe was provided.")
        x = x.merge(
            sentiment_df[["Date", "Ticker", "sentiment_t1", "sentiment_conf_t1"]],
            on=["Date", "Ticker"],
            how="left",
        )
        x["sentiment_t1"] = x["sentiment_t1"].fillna(0.0)
        x["sentiment_conf_t1"] = x["sentiment_conf_t1"].fillna(0.0)

    if cfg.use_momentum and cfg.use_finbert_sentiment:
        # DL-enhanced score without explicit HMM state uses weighted blend.
        x["predicted_alpha"] = 0.6 * x["momentum_t1"] + 0.4 * x["sentiment_t1"]
    elif cfg.use_momentum:
        x["predicted_alpha"] = x["momentum_t1"]
    elif cfg.use_finbert_sentiment:
        x["predicted_alpha"] = x["sentiment_t1"]
    else:
        raise ValueError("Invalid toggle combination: at least one signal source must be enabled.")

    out = x[["Date", "Ticker", "predicted_alpha", "actual_alpha"]].copy()
    out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return out


def run_ablation_backtest(
    config: AblationConfig,
    *,
    processed_dir: Path,
    nifty_features_path: Path,
    raw_dir: Path,
    start: str,
    end: str,
    sentiment_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _load_momentum_panel(processed_dir=processed_dir, start=start, end=end)

    sentiment_df = None
    if config.use_finbert_sentiment:
        if sentiment_path is None:
            raise ValueError("sentiment_path is required when use_finbert_sentiment=True")
        sentiment_df = _load_sentiment(sentiment_path)

    preds = _build_predictions(panel, config, sentiment_df)

    bt = run_backtest_daily(
        predictions_df=preds,
        nifty_features_path=str(nifty_features_path),
        raw_dir=str(raw_dir),
        start=start,
        end=end,
        top_n=int(config.top_n),
        capital=float(config.capital),
        stop_loss=0.07,
        stop_loss_bear=0.05,
        max_weight=0.25,
        portfolio_dd_cash_trigger=0.15,
        rebalance_friction_rate=0.0022,
        turnover_penalty_score=0.0,
        kelly_p=0.5238,
        kelly_use_predicted_margin=True,
        kelly_scale_position=True,
        use_sector_cap=True,
        flat_regime_sizing=not bool(config.use_hmm_regime),
    )
    return preds, bt


def compute_metrics(bt: pd.DataFrame, capital: float) -> dict[str, float]:
    if bt.empty or len(bt) < 2:
        return {
            "CAGR": 0.0,
            "Sharpe_Net": 0.0,
            "Max_Drawdown": 0.0,
            "Win_Rate": 0.0,
        }

    r = bt["portfolio_return"].astype(float).fillna(0.0)
    nav = bt["portfolio_value"].astype(float).to_numpy()

    start_ts = pd.to_datetime(bt["date"].iloc[0])
    end_ts = pd.to_datetime(bt["date"].iloc[-1])
    years = max((end_ts - start_ts).days / 365.25, 1e-9)

    total_ret = float(nav[-1] / float(capital) - 1.0)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0

    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(12.0))

    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())

    win_rate = float((r > 0).mean())

    return {
        "CAGR": cagr,
        "Sharpe_Net": sharpe,
        "Max_Drawdown": max_dd,
        "Win_Rate": win_rate,
    }


def sentiment_alpha_by_regime(with_sent: pd.DataFrame, without_sent: pd.DataFrame) -> pd.DataFrame:
    if with_sent.empty or without_sent.empty:
        return pd.DataFrame(columns=["regime", "sentiment_alpha_mean_return", "sentiment_alpha_total_return"])

    a = with_sent[["date", "regime", "portfolio_return"]].rename(columns={"portfolio_return": "ret_with"})
    b = without_sent[["date", "regime", "portfolio_return"]].rename(columns={"portfolio_return": "ret_without"})

    m = a.merge(b, on=["date", "regime"], how="inner")
    if m.empty:
        return pd.DataFrame(columns=["regime", "sentiment_alpha_mean_return", "sentiment_alpha_total_return"])

    m["ret_delta"] = m["ret_with"] - m["ret_without"]
    out = (
        m.groupby("regime", as_index=False)
        .agg(
            sentiment_alpha_mean_return=("ret_delta", "mean"),
            sentiment_alpha_total_return=("ret_delta", "sum"),
            windows=("ret_delta", "count"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )
    return out
