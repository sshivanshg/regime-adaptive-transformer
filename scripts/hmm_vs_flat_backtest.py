"""
Compare two portfolio rules on the same ranking file:

  1) HMM-conditioned — default ``run_backtest_daily`` (NIFTY ``HMM_Regime`` drives
     sleeve size and top-N by regime).
  2) Regime-agnostic (flat sizing) — ``flat_regime_sizing=True``: full sleeve
     1.0/1.0/1.0 and ``top_n`` in every regime (no HMM-based *portfolio* scaling).

This is **not** “train RAMT without HMM in the model”; predictions are identical.

Writes under ``results/hmm_vs_flat/`` only (does not overwrite ``backtest_results.csv``).

Examples::

  # Requires predictions + NIFTY features covering the window (this repo: ~2020+).
  python scripts/hmm_vs_flat_backtest.py --start 2008-01-01 --end 2010-12-31

  python scripts/hmm_vs_flat_backtest.py --start 2024-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.backtest import run_backtest_daily  # noqa: E402


def compute_metrics(bt: pd.DataFrame, capital: float) -> dict[str, float]:
    r = bt["portfolio_return"].dropna()
    if bt.empty or len(r) < 2:
        return {}
    nav = bt["portfolio_value"].values.astype(float)
    start_ts = bt["date"].iloc[0]
    end_ts = bt["date"].iloc[-1]
    span_years = (end_ts - start_ts).days / 365.25

    sharpe_net = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0
    total_ret = nav[-1] / capital - 1.0
    cagr = (1 + total_ret) ** (1 / span_years) - 1 if span_years > 0 else 0.0
    peak = np.maximum.accumulate(nav)
    max_dd = float(((nav - peak) / peak).min())

    return {
        "sharpe": sharpe_net,
        "cagr_pct": cagr * 100.0,
        "max_dd_pct": max_dd * 100.0,
        "total_return_pct": total_ret * 100.0,
        "n_windows": float(len(bt)),
        "span_years": float(span_years),
    }


def portfolio_kwargs(*, flat_regime_sizing: bool) -> dict[str, Any]:
    return {
        "capital": 100_000,
        "top_n": 5,
        "stop_loss": 0.07,
        "stop_loss_bear": 0.05,
        "max_weight": 0.25,
        "portfolio_dd_cash_trigger": 0.15,
        "rebalance_friction_rate": 0.002,
        "turnover_penalty_score": 0.0,
        "kelly_p": 0.5238,
        "kelly_use_predicted_margin": True,
        "kelly_scale_position": True,
        "use_sector_cap": True,
        "flat_regime_sizing": flat_regime_sizing,
    }


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")[:80]


def main() -> None:
    p = argparse.ArgumentParser(
        description="HMM-conditioned vs flat regime sizing (same predictions)"
    )
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument(
        "--predictions",
        type=str,
        default=str(ROOT / "results" / "ranking_predictions.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "hmm_vs_flat"),
    )
    p.add_argument(
        "--include-train",
        action="store_true",
        help="Use all Period rows in range (default: Test only)",
    )
    p.add_argument(
        "--raw-dir",
        type=str,
        default=str(ROOT / "data" / "raw"),
        help="Parquet OHLCV folder (must cover tickers in predictions)",
    )
    p.add_argument(
        "--nifty-features",
        type=str,
        default=str(ROOT / "data" / "processed" / "_NSEI_features.parquet"),
        help="Processed NIFTY features with HMM_Regime column",
    )
    args = p.parse_args()

    preds_path = Path(args.predictions)
    if not preds_path.is_file():
        raise SystemExit(f"Predictions file not found: {preds_path.resolve()}")

    preds = pd.read_csv(preds_path)
    preds["Date"] = pd.to_datetime(preds["Date"])
    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)

    if "Period" in preds.columns and not args.include_train:
        preds = preds[preds["Period"] == "Test"]
    sub = preds[(preds["Date"] >= start_ts) & (preds["Date"] <= end_ts)][
        ["Date", "Ticker", "predicted_alpha", "actual_alpha"]
    ].copy()

    if sub.empty:
        dmin, dmax = preds["Date"].min(), preds["Date"].max()
        raise SystemExit(
            f"No prediction rows in [{args.start}, {args.end}] "
            f"(after Period filter: {'Test only' if 'Period' in preds.columns and not args.include_train else 'none'}). "
            f"File date span: {dmin.date()} … {dmax.date()}. "
            f"Extend features/raw data and regenerate rankings, or pick a window inside that span."
        )

    out_root = Path(args.out_dir)
    label = f"{args.start}_{args.end}"
    out_dir = out_root / slug(label)
    out_dir.mkdir(parents=True, exist_ok=True)

    nifty_features = str(Path(args.nifty_features).resolve())
    raw_dir = str(Path(args.raw_dir).resolve())

    rows = []
    for name, flat in (
        ("hmm_conditioned_portfolio", False),
        ("regime_agnostic_flat_sizing", True),
    ):
        bt = run_backtest_daily(
            predictions_df=sub,
            nifty_features_path=nifty_features,
            raw_dir=raw_dir,
            start=args.start,
            end=args.end,
            **portfolio_kwargs(flat_regime_sizing=flat),
        )
        path = (out_dir / f"backtest_{name}.csv").resolve()
        bt.to_csv(path, index=False)
        m = compute_metrics(bt, 100_000.0)
        base_row: dict[str, Any] = {
            "study_name": "HMM-conditioned vs regime-agnostic (flat) portfolio sizing",
            "variant": name,
            "description": (
                "NIFTY HMM drives sleeves and top-N"
                if not flat
                else "flat 1.0/1.0/1.0, top_n all regimes"
            ),
            "window_start": args.start,
            "window_end": args.end,
            "n_prediction_rows": len(sub),
            "predictions_file": str(preds_path.resolve()),
            "raw_dir": raw_dir,
            "nifty_features": nifty_features,
            "backtest_csv": str(path.relative_to(ROOT.resolve())),
        }
        if m:
            base_row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        else:
            base_row["note"] = "empty backtest output"
        rows.append(base_row)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "hmm_vs_flat_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "terminology": {
            "hmm_conditioned": (
                "Portfolio uses NIFTY HMM_Regime: bull / high-vol / bear sleeves "
                "(e.g. 100% / 50% / 20%) and regime-specific top-N."
            ),
            "regime_agnostic_flat": (
                "Portfolio ignores HMM for sizing: full sleeve every regime (flat_regime_sizing). "
                "Same ranking inputs as the HMM run."
            ),
        },
        "historical_yfinance": (
            "For pre-2020 windows use: scripts/fetch_nifty200.py --start --end --raw-dir, "
            "scripts/build_processed_range.py, scripts/momentum_predictions_from_features.py, "
            "then pass --raw-dir and --nifty-features here."
        ),
    }
    with open(out_dir / "hmm_vs_flat_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(summary.to_string(index=False))
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
