"""
Weekly momentum experiment (2023–2026) using the same production backtest path.

This is intentionally a *quick experiment*:
- Build a `ranking_predictions`-style CSV using pure 21d momentum (`Ret_21d`)
  as `predicted_alpha` and `Sector_Alpha` as `actual_alpha`.
- Rebalance every 5th NIFTY trading day (≈ weekly).
- Run the existing `models.backtest.run_backtest_daily` to get an apples-to-apples
  NAV series with friction, sector cap, HMM regime sizing, and stops.

Outputs (under `results/final_strategy/`):
- `weekly_ranking_predictions_2023_2026.csv`
- `backtest_results_weekly_2023_2026.csv`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.backtest import run_backtest_daily  # noqa: E402


def build_weekly_predictions(
    *,
    processed_dir: Path,
    start: str,
    end: str,
    rebalance_every: int,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Create a ranking_predictions-style file from processed features.
    """
    processed = processed_dir.resolve()
    nifty_path = processed / "_NSEI_features.parquet"
    if not nifty_path.is_file():
        raise SystemExit(f"Missing NIFTY features parquet: {nifty_path}")

    paths = sorted(
        p
        for p in processed.glob("*_features.parquet")
        if p.name != "_NSEI_features.parquet" and not p.stem.startswith("_")
    )
    if not paths:
        raise SystemExit(f"No stock feature files in {processed}")

    chunks: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(
            p,
            columns=["Date", "Ticker", "Ret_21d", "Sector_Alpha"],
        )
        chunks.append(df)
    panel = pd.concat(chunks, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Use NIFTY trading calendar to define rebalance dates
    n = pd.read_parquet(nifty_path, columns=["Date"])
    n["Date"] = pd.to_datetime(n["Date"])
    nd = n["Date"].sort_values().unique()
    nd = nd[(nd >= start_ts) & (nd <= end_ts)]
    if len(nd) == 0:
        raise SystemExit("No NIFTY trading days in requested window.")
    rebal = nd[:: int(rebalance_every)]
    rebal_set = set(pd.Timestamp(x) for x in rebal)

    panel = panel[(panel["Date"] >= start_ts) & (panel["Date"] <= end_ts)]
    panel = panel[panel["Date"].isin(rebal_set)]
    panel = panel.dropna(subset=["Ret_21d"])

    out = panel.rename(
        columns={
            "Ret_21d": "predicted_alpha",
            "Sector_Alpha": "actual_alpha",
        }
    )
    out["actual_alpha"] = pd.to_numeric(out["actual_alpha"], errors="coerce").fillna(0.0)
    out["predicted_alpha"] = pd.to_numeric(out["predicted_alpha"], errors="coerce").fillna(
        0.0
    )
    out["Period"] = "Test"

    out = out[["Date", "Ticker", "predicted_alpha", "actual_alpha", "Period"]].sort_values(
        ["Date", "predicted_alpha"], ascending=[True, False]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2023-01-01")
    ap.add_argument("--end", type=str, default="2026-04-16")
    ap.add_argument("--rebalance-every", type=int, default=5)
    ap.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    out_pred = (
        ROOT
        / "results"
        / "final_strategy"
        / "weekly_ranking_predictions_2023_2026.csv"
    )
    out_bt = (
        ROOT
        / "results"
        / "final_strategy"
        / "backtest_results_weekly_2023_2026.csv"
    )

    preds = build_weekly_predictions(
        processed_dir=Path(args.processed_dir),
        start=str(args.start),
        end=str(args.end),
        rebalance_every=int(args.rebalance_every),
        out_csv=out_pred,
    )

    bt = run_backtest_daily(
        predictions_df=preds[["Date", "Ticker", "predicted_alpha", "actual_alpha"]],
        nifty_features_path=str(ROOT / "data/processed/_NSEI_features.parquet"),
        raw_dir=str(ROOT / "data/raw"),
        start=str(args.start),
        end=str(args.end),
        top_n=5,
        capital=100000,
        stop_loss=0.07,
        max_weight=0.25,
        portfolio_dd_cash_trigger=0.15,
        rebalance_friction_rate=0.002,
        turnover_penalty_score=0.0,
        kelly_p=0.5238,
        kelly_use_predicted_margin=True,
        kelly_scale_position=True,
        use_sector_cap=True,
        flat_regime_sizing=False,
    )
    out_bt.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out_bt, index=False)

    print(f"Wrote predictions: {out_pred}")
    print(f"Wrote backtest:    {out_bt}")


if __name__ == "__main__":
    main()

