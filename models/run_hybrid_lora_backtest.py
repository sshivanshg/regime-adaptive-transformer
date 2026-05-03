"""
User Story:
As a quant researcher, I need to run a backtest that integrates LoRA sentiment with
RAMT momentum and HMM regime detection to generate the true "Full Hybrid" result.

Implementation Approach:
Load RAMT predictions (momentum scores), LoRA predictions (sentiment scores),
and HMM regimes. Use build_hybrid_rankings to create sentiment-gated rankings,
then run the backtest with risk controls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.backtest import run_backtest_daily
from models.hybrid_strategy import build_hybrid_rankings, HybridRuleConfig
from features.feature_engineering import _safe_stem_from_ticker


def normalize_ticker(ticker: str) -> str:
    return ticker.upper().strip().replace(".", "_")


def load_lora_sentiment(lora_path: Path, date_filter_start: str = "2024-01-01") -> pd.DataFrame:
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA predictions not found: {lora_path}")

    df = pd.read_csv(lora_path, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].apply(normalize_ticker)

    # Normalize predicted values to sentiment-like range (-1 to 1)
    # LoRA predicts alpha, so we z-score and clip
    pred_mean = df["predicted"].mean()
    pred_std = df["predicted"].std()
    df["sentiment_score"] = np.clip((df["predicted"] - pred_mean) / (pred_std + 1e-8), -1, 1)
    df["sentiment_confidence"] = np.abs(df["sentiment_score"])

    # Filter to backtest period
    df = df[df["Date"] >= pd.to_datetime(date_filter_start)]

    return df[["Date", "Ticker", "sentiment_score", "sentiment_confidence"]]


def load_rampt_predictions(pred_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"RAMT predictions not found: {pred_path}")

    df = pd.read_csv(pred_path, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].apply(normalize_ticker)

    # Keep only Test period
    df = df[df["Period"] == "Test"]

    return df[["Date", "Ticker", "predicted_alpha", "actual_alpha"]]


def get_hmm_regimes(nifty_features_path: Path) -> pd.DataFrame:
    if not nifty_features_path.exists():
        raise FileNotFoundError(f"NIFTY features not found: {nifty_features_path}")

    df = pd.read_parquet(nifty_features_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    if "HMM_Regime" not in df.columns:
        raise ValueError("NIFTY features missing 'HMM_Regime' column")

    reg_df = df[["Date", "HMM_Regime"]].drop_duplicates().sort_values("Date").copy()
    reg_df = reg_df.rename(columns={"HMM_Regime": "regime"})
    return reg_df


def add_momentum_to_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Add 21-day momentum to rankings for display purposes."""
    if rankings.empty:
        return rankings

    cache: dict[str, pd.DataFrame] = {}
    moms = []

    for _, row in rankings.iterrows():
        d = pd.to_datetime(row["Date"])
        t = row["Ticker"]
        if t == "CASH":
            moms.append(np.nan)
            continue
        if t not in cache:
            stem = _safe_stem_from_ticker(str(t))
            p = ROOT / "data/processed" / f"{stem}_features.parquet"
            if not p.exists():
                moms.append(np.nan)
                continue
            df = pd.read_parquet(p)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            cache[t] = df
        df = cache[t]
        sub = df.loc[:d]
        if sub.empty:
            moms.append(np.nan)
            continue
        last = sub.iloc[-1]
        moms.append(float(last["Ret_21d"]) if "Ret_21d" in last else np.nan)

    rankings = rankings.copy()
    rankings["momentum_t1"] = moms
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest with Hybrid (RAMT + LoRA + HMM)")
    parser.add_argument("--predictions", type=str, default="results/final_strategy/ranking_predictions.csv",
                        help="RAMT predictions CSV")
    parser.add_argument("--lora", type=str, default="results/lora/lora_predictions.csv",
                        help="LoRA predictions CSV")
    parser.add_argument("--nifty-features", type=str, default="data/processed/_NSEI_features.parquet",
                        help="NIFTY features parquet for HMM regimes")
    parser.add_argument("--output", type=str, default="results/hybrid_lora/backtest_results.csv",
                        help="Output path")
    parser.add_argument("--top-n", type=int, default=5, help="Number of stocks to pick")
    parser.add_argument("--start", type=str, default="2008-01-01", help="Backtest start date")
    parser.add_argument("--end", type=str, default="2012-12-31", help="Backtest end date")
    args = parser.parse_args()

    print("Loading data...")
    pred_path = ROOT / args.predictions
    lora_path = ROOT / args.lora
    nifty_path = ROOT / args.nifty_features

    momentum_df = load_rampt_predictions(pred_path)
    sentiment_df = load_lora_sentiment(lora_path, "2024-01-01")
    regime_df = get_hmm_regimes(nifty_path)

    print(f"  RAMT predictions: {len(momentum_df)} rows")
    print(f"  LoRA sentiment: {len(sentiment_df)} rows")
    print(f"  HMM regimes: {len(regime_df)} rows")

    print("Building hybrid rankings (RAMT + LoRA + HMM)...")
    config = HybridRuleConfig(
        top_n=args.top_n,
        bull_sentiment_floor=-0.2,
        bear_sentiment_floor=0.5,
        high_vol_momentum_weight=0.6,
        high_vol_sentiment_weight=0.4,
    )

    hybrid_df = build_hybrid_rankings(
        momentum_df=momentum_df,
        sentiment_df=sentiment_df,
        regime_df=regime_df,
        config=config,
        momentum_col="predicted_alpha",
    )

    # Filter out CASH rows for backtest - they will be handled by position sizing
    hybrid_df = hybrid_df[hybrid_df["Ticker"] != "CASH"].copy()

    # Add actual_alpha from original momentum data
    alpha_map = momentum_df[["Date", "Ticker", "actual_alpha"]].copy()
    alpha_map = alpha_map.rename(columns={"actual_alpha": "actual_alpha_src"})
    # Normalize ticker for merge
    alpha_map["Ticker_norm"] = alpha_map["Ticker"].apply(normalize_ticker)
    hybrid_df["Ticker_norm"] = hybrid_df["Ticker"]
    hybrid_df = hybrid_df.merge(alpha_map[["Date", "Ticker_norm", "actual_alpha_src"]], 
                                  on=["Date", "Ticker_norm"], how="left")
    hybrid_df = hybrid_df.rename(columns={"actual_alpha_src": "actual_alpha"})

    # Convert back to original ticker format for backtest
    hybrid_df["Ticker_orig"] = hybrid_df["Ticker"].str.replace("_", ".", regex=False)

    hybrid_df = add_momentum_to_rankings(hybrid_df)

    # Prepare for backtest
    pred_for_bt = hybrid_df[["Date", "Ticker_orig", "hybrid_score", "actual_alpha"]].copy()
    pred_for_bt = pred_for_bt.rename(columns={"Ticker_orig": "Ticker"})
    pred_for_bt = pred_for_bt.rename(columns={"hybrid_score": "predicted_alpha"})

    out_dir = ROOT / "results" / "hybrid_lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running backtest with risk controls...")
    bt = run_backtest_daily(
        predictions_df=pred_for_bt,
        nifty_features_path=str(nifty_path),
        raw_dir=str(ROOT / "data/raw"),
        start="2024-01-01",
        end="2026-04-16",
        top_n=args.top_n,
        capital=100000,
        stop_loss=0.07,
        max_weight=0.25,
        portfolio_dd_cash_trigger=0.15,
        rebalance_friction_rate=0.002,
        turnover_penalty_score=0.0,
        kelly_p=0.5238,
        kelly_use_predicted_margin=True,
        kelly_scale_position=True,
    )

    bt_out = ROOT / args.output
    bt.to_csv(bt_out, index=False)
    print(f"Saved backtest results: {bt_out}")

    # Calculate summary metrics
    capital = 100000
    start_val = bt["portfolio_value_start"].iloc[0]
    end_val = bt["portfolio_value"].iloc[-1]
    years = (pd.to_datetime(bt["date"].iloc[-1]) - pd.to_datetime(bt["date"].iloc[0])).days / 365.25

    total_ret = (end_val / capital) - 1
    cagr = (1 + total_ret) ** (1 / years) - 1

    returns = bt["portfolio_return"].astype(float)
    sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(12)

    peak = np.maximum.accumulate(bt["portfolio_value"].values)
    max_dd = ((bt["portfolio_value"].values - peak) / peak).min()

    win_rate = (returns > 0).mean()

    print("\n" + "=" * 50)
    print("HYBRID (RAMT + LoRA + HMM) BACKTEST RESULTS")
    print("=" * 50)
    print(f"Period: {bt['date'].iloc[0]} to {bt['date'].iloc[-1]}")
    print(f"CAGR:        {cagr*100:.2f}%")
    print(f"Sharpe:      {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Win Rate:    {win_rate*100:.1f}%")
    print(f"Total Return: {total_ret*100:.2f}%")
    print("=" * 50)

    # Also save hybrid rankings
    rank_out = out_dir / "hybrid_rankings.csv"
    hybrid_df.to_csv(rank_out, index=False)
    print(f"Saved hybrid rankings: {rank_out}")


if __name__ == "__main__":
    main()