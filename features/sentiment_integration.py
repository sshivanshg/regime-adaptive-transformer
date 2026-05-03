from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SentimentMergeConfig:
    locf_max_days: int = 3
    sentiment_momentum_lookback: int = 5
    fill_missing_with_neutral: bool = True


def _normalize_ticker(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.upper()
    t = t.str.replace("_", ".", regex=False)
    t = t.str.replace(".NS", "_NS", regex=False)
    return t


def prepare_sentiment_daily(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"Date", "Ticker", "sentiment_score", "sentiment_confidence"}
    missing = sorted(needed.difference(sentiment_df.columns))
    if missing:
        raise ValueError(f"Sentiment frame missing columns: {missing}")

    out = sentiment_df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out["Ticker"] = _normalize_ticker(out["Ticker"])

    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce")
    out["sentiment_confidence"] = pd.to_numeric(out["sentiment_confidence"], errors="coerce")

    # If multiple rows exist for a date/ticker, keep confidence-weighted aggregate.
    out["_w"] = out["sentiment_confidence"].clip(lower=0.0)
    out["_num"] = out["sentiment_score"].fillna(0.0) * out["_w"]
    agg = (
        out.groupby(["Date", "Ticker"], as_index=False)
        .agg(
            _num=("_num", "sum"),
            _den=("_w", "sum"),
            sentiment_confidence=("sentiment_confidence", "mean"),
            sentiment_score_mean=("sentiment_score", "mean"),
        )
    )
    agg["sentiment_score"] = np.where(
        agg["_den"] > 1e-12,
        agg["_num"] / agg["_den"],
        agg["sentiment_score_mean"].fillna(0.0),
    )
    agg = agg[["Date", "Ticker", "sentiment_score", "sentiment_confidence"]]
    return agg


def merge_sentiment_features(
    features_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    config: SentimentMergeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SentimentMergeConfig()

    needed = {"Date", "Ticker"}
    missing = sorted(needed.difference(features_df.columns))
    if missing:
        raise ValueError(f"Features frame missing columns: {missing}")

    base = features_df.copy()
    base["Date"] = pd.to_datetime(base["Date"]).dt.tz_localize(None)
    base["Ticker"] = _normalize_ticker(base["Ticker"])

    s_daily = prepare_sentiment_daily(sentiment_df)

    merged = base.merge(s_daily, on=["Date", "Ticker"], how="left", sort=False)
    merged = merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # LOCF with explicit stale cap (3 days by default) to avoid stale narratives.
    merged["sentiment_score"] = merged.groupby("Ticker")["sentiment_score"].ffill(limit=int(cfg.locf_max_days))
    merged["sentiment_confidence"] = merged.groupby("Ticker")["sentiment_confidence"].ffill(
        limit=int(cfg.locf_max_days)
    )

    if cfg.fill_missing_with_neutral:
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
        merged["sentiment_confidence"] = merged["sentiment_confidence"].fillna(0.0)

    lookback = int(cfg.sentiment_momentum_lookback)
    merged["sentiment_momentum"] = merged.groupby("Ticker")["sentiment_score"].diff(lookback)
    merged["sentiment_momentum"] = merged["sentiment_momentum"].fillna(0.0)

    return merged
