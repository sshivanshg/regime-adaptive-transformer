"""
Test 3 — Parameter sensitivity: rerun run_backtest_daily with varied hyperparameters.

Reads predictions from the repo (default: results/ranking_predictions.csv) and writes
ONLY under results/sensitivity/ — does not overwrite results/backtest_results.csv.

Usage (from repo root):
  python scripts/parameter_sensitivity_backtest.py
  python scripts/parameter_sensitivity_backtest.py --predictions path/to/ranking_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.backtest import run_backtest_daily  # noqa: E402


def compute_metrics(bt: pd.DataFrame, capital: float) -> dict[str, float]:
    """Match dashboard/app.py compute_metrics on daily backtest rows."""
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

    return {
        "sharpe": sharpe_net,
        "cagr_pct": cagr * 100.0,
        "max_dd_pct": max_dd * 100.0,
        "total_return_pct": total_ret * 100.0,
        "n_windows": float(len(bt)),
        "span_years": float(span_years),
    }


def baseline_kwargs() -> dict[str, Any]:
    """Same portfolio rules as models/run_final_2024_2026.py backtest block."""
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
        "flat_regime_sizing": False,
    }


def variants() -> list[tuple[str, dict[str, Any]]]:
    b = baseline_kwargs()
    return [
        (
            "Baseline (top-5, SL=7%, sector cap on, regime sizing on)",
            {**b},
        ),
        ("top-3", {**b, "top_n": 3}),
        ("top-7", {**b, "top_n": 7}),
        ("SL=5%", {**b, "stop_loss": 0.05}),
        ("SL=10%", {**b, "stop_loss": 0.10}),
        ("No sector cap", {**b, "use_sector_cap": False}),
        ("No regime sizing (flat 1.0/1.0/1.0)", {**b, "flat_regime_sizing": True}),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sensitivity grid for run_backtest_daily")
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(ROOT / "results" / "ranking_predictions.csv"),
        help="Path to ranking_predictions.csv (read-only)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "sensitivity"),
        help="Output directory (default: results/sensitivity)",
    )
    args = parser.parse_args()

    preds_path = Path(args.predictions)
    if not preds_path.is_file():
        raise SystemExit(f"Predictions file not found: {preds_path.resolve()}")

    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    test_start = "2024-01-01"
    test_end = "2026-04-16"
    nifty_features = str(ROOT / "data" / "processed" / "_NSEI_features.parquet")
    raw_dir = str(ROOT / "data" / "raw")

    preds = pd.read_csv(preds_path)
    preds["Date"] = pd.to_datetime(preds["Date"])
    test_df = preds[preds["Period"] == "Test"][
        ["Date", "Ticker", "predicted_alpha", "actual_alpha"]
    ].copy()

    rows: list[dict[str, Any]] = []
    for label, kw in variants():
        bt = run_backtest_daily(
            predictions_df=test_df,
            nifty_features_path=nifty_features,
            raw_dir=raw_dir,
            start=test_start,
            end=test_end,
            **kw,
        )
        if bt.empty:
            rows.append(
                {
                    "variant": label,
                    "sharpe": float("nan"),
                    "cagr_pct": float("nan"),
                    "max_dd_pct": float("nan"),
                    "note": "empty backtest",
                }
            )
            continue
        m = compute_metrics(bt, float(kw["capital"]))
        safe_name = (
            label.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace("%", "pct")
            .replace("/", "-")[:80]
        )
        bt_path = runs_dir / f"backtest_{safe_name}.csv"
        bt.to_csv(bt_path, index=False)
        rows.append(
            {
                "variant": label,
                "sharpe": round(m["sharpe"], 4),
                "cagr_pct": round(m["cagr_pct"], 2),
                "max_dd_pct": round(m["max_dd_pct"], 2),
                "total_return_pct": round(m["total_return_pct"], 2),
                "n_windows": int(m["n_windows"]),
                "backtest_csv": str(bt_path.relative_to(ROOT)),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "parameter_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "predictions_source": str(preds_path.resolve()),
        "test_window": [test_start, test_end],
        "metrics_note": "Sharpe/CAGR/MaxDD aligned with dashboard: sqrt(12) on window returns; CAGR from span_years",
        "stop_loss_caveat": (
            "In models/backtest.py run_backtest_daily, per-stock stops set sl_stock and flags "
            "stops_hit but realized window returns still use the full price path "
            "(_window_period_return). So varying stop_loss alone may duplicate metrics until "
            "stop-outs are applied to returns."
        ),
    }
    with open(out_dir / "parameter_sensitivity_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(summary.to_string(index=False))
    print(f"\nWrote: {summary_path}")
    print(f"Per-variant CSVs under: {runs_dir}")


if __name__ == "__main__":
    main()
