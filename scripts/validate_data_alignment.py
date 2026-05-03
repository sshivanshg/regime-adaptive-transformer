"""
User Story:
Prevent invalid backtests by ensuring price, HMM regime, and sentiment dates are
perfectly aligned over the requested evaluation window.

Implementation Approach:
Load all three data sources, compute their date sets in-window, compare exact set
equality, and emit a detailed missing-date report before any portfolio simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_alignment(
    sentiment_path: Path,
    processed_dir: Path,
    nifty_features_path: Path,
    start: str,
    end: str,
) -> tuple[bool, dict[str, object]]:
    s = pd.read_parquet(sentiment_path)
    s["Date"] = pd.to_datetime(s["Date"]).dt.tz_localize(None)
    s_dates = set(s[(s["Date"] >= pd.to_datetime(start)) & (s["Date"] <= pd.to_datetime(end))]["Date"].unique())

    n = pd.read_parquet(nifty_features_path, columns=["Date", "HMM_Regime"])
    n["Date"] = pd.to_datetime(n["Date"]).dt.tz_localize(None)
    hmm_dates = set(n[(n["Date"] >= pd.to_datetime(start)) & (n["Date"] <= pd.to_datetime(end))]["Date"].unique())

    price_dates: set[pd.Timestamp] = set()
    files = sorted(
        p
        for p in processed_dir.glob("*_features.parquet")
        if p.name != "_NSEI_features.parquet"
    )
    for p in files[:20]:
        d = pd.read_parquet(p, columns=["Date"])
        d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
        ds = set(d[(d["Date"] >= pd.to_datetime(start)) & (d["Date"] <= pd.to_datetime(end))]["Date"].unique())
        if not price_dates:
            price_dates = ds
        else:
            price_dates = price_dates.intersection(ds)

    common = sorted(price_dates.intersection(hmm_dates).intersection(s_dates))
    ok = (len(common) > 0) and (s_dates == hmm_dates == price_dates)

    report = {
        "ok": bool(ok),
        "n_dates_common": int(len(common)),
        "n_dates_sentiment": int(len(s_dates)),
        "n_dates_hmm": int(len(hmm_dates)),
        "n_dates_price_intersection": int(len(price_dates)),
        "missing_in_sentiment": int(len((price_dates.intersection(hmm_dates)) - s_dates)),
        "missing_in_hmm": int(len((price_dates.intersection(s_dates)) - hmm_dates)),
        "missing_in_price": int(len((s_dates.intersection(hmm_dates)) - price_dates)),
    }
    return ok, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate sentiment/price/HMM date alignment before backtest.")
    parser.add_argument("--config", type=str, default="config/hybrid_config.yaml")
    parser.add_argument("--sentiment-path", type=str, default=None)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-04-16")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    paths = cfg.get("paths", {})

    sentiment_path = Path(args.sentiment_path or paths.get("sentiment_output", "data/processed/sentiment/sentiment_features_lora.parquet"))
    processed_dir = Path(paths.get("processed_features_dir", "data/processed"))
    nifty_features_path = Path(paths.get("nifty_features", "data/processed/_NSEI_features.parquet"))

    ok, rep = validate_alignment(sentiment_path, processed_dir, nifty_features_path, args.start, args.end)

    print("Alignment report:")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    if not ok:
        raise SystemExit("Alignment validation failed. Please regenerate missing data before backtesting.")


if __name__ == "__main__":
    main()
