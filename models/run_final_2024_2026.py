"""
Final split runner (no future leakage)

Train: 2015-01-01 … 2022-12-31 (history used to fit scalers + model).
Blind test / backtest: 2023-01-01 … 2026-04-15 (held out from scaler fitting).

Walk-forward: **retrain** every ``training_step`` (default 126 trading days ≈ 6 months);
**inference / portfolio rebalance** every ``rebalance_step`` (default 21 days). Inference
uses a 30-day warm-up requirement beyond the model sequence length for stable indicators.

Produces:
- results/ranking_predictions.csv
- results/monthly_rankings.csv
- results/backtest_results.csv
- results/ramt_model_state.pt + scaler artifacts refreshed after every walk-forward fold
- results/training_history.csv & results/training_dashboard.png (unless --no-plots)

Use ``--backtest-only`` to skip training and reuse ``ranking_predictions.csv`` after a full run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.backtest import run_backtest_daily  # noqa: E402
from models.ramt import train_ranking as tr  # noqa: E402


def add_momentum_column(rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a momentum score for dashboard display.
    For the NIFTY200 Parquet pipeline, we use Ret_21d if available (recent momentum proxy).
    """
    if rankings.empty:
        rankings["momentum"] = []
        return rankings

    # Lazy per-ticker cache of processed feature frames
    cache: dict[str, pd.DataFrame] = {}
    moms = []
    for _, row in rankings.iterrows():
        d = pd.to_datetime(row["Date"])
        t = row["Ticker"]
        if t not in cache:
            p = ROOT / "data/processed" / f"{t}_features.parquet"
            df = pd.read_parquet(p)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            cache[t] = df
        df = cache[t]
        # last available on/before date
        sub = df.loc[:d]
        if sub.empty:
            moms.append(0.0)
            continue
        last = sub.iloc[-1]
        moms.append(float(last["Ret_21d"]) if "Ret_21d" in last else 0.0)

    rankings = rankings.copy()
    rankings["momentum"] = moms
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="RAMT blind-split final run (2023–2026 test)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs for combined_walk_forward (default: 3)",
    )
    parser.add_argument(
        "--training-step",
        type=int,
        default=126,
        help="Walk-forward retrain cadence in trading days (~6 months). Default: 126",
    )
    parser.add_argument(
        "--rebalance-step",
        type=int,
        default=21,
        help="Inference and portfolio rebalance cadence inside each OOS segment (default: 21)",
    )
    parser.add_argument(
        "--inference-warmup-days",
        type=int,
        default=30,
        help="Extra history rows required before scoring (SEQ_LEN + warmup). Default: 30",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Deprecated: sets rebalance-step if provided (for old scripts)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override tr.BATCH_SIZE (default: keep train_ranking default)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience on val loss (default: train_ranking.PATIENCE)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib dashboard + training_history.csv export",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Skip training; load predictions CSV and run rankings + backtest only",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to ranking_predictions.csv for --backtest-only (default: results/ranking_predictions.csv)",
    )
    args = parser.parse_args()

    os.makedirs(ROOT / "results", exist_ok=True)

    # Strict blind split (matches train_ranking.py)
    train_start = "2015-01-01"
    train_end = "2022-12-31"
    test_start = "2023-01-01"
    test_end = "2026-04-15"

    preds_path = (
        Path(args.predictions)
        if args.predictions
        else ROOT / "results" / "ranking_predictions.csv"
    )

    if args.backtest_only:
        if not preds_path.is_file():
            raise SystemExit(f"--backtest-only: predictions file not found: {preds_path.resolve()}")
        print(f"Loading predictions (no training): {preds_path.resolve()}", flush=True)
        rs = args.rebalance_step if args.step_size is None else int(args.step_size)
        print(
            f"Using rebalance_step={rs} for backtest — match the step used when "
            f"ranking_predictions.csv was generated.",
            flush=True,
        )
        preds = pd.read_csv(preds_path)
        preds["Date"] = pd.to_datetime(preds["Date"])
        required = {"Date", "Ticker", "predicted_alpha", "actual_alpha", "Period"}
        missing = required - set(preds.columns)
        if missing:
            raise SystemExit(f"Predictions CSV missing columns {missing}; need {required}")
    else:
        tr.PATIENCE = args.patience if args.patience is not None else tr.PATIENCE
        if args.batch_size is not None:
            tr.BATCH_SIZE = int(args.batch_size)

        plot_dir = None if args.no_plots else str(ROOT / "results")

        rebalance_step = int(args.rebalance_step) if args.step_size is None else int(args.step_size)
        print(
            f"Training WF model (epochs={args.epochs}, training_step={args.training_step}, "
            f"rebalance_step={rebalance_step}, warmup={args.inference_warmup_days})…",
            flush=True,
        )
        preds = tr.combined_walk_forward(
            start=train_start,
            end=test_end,
            training_step=int(args.training_step),
            rebalance_step=rebalance_step,
            inference_warmup_days=int(args.inference_warmup_days),
            max_epochs=int(args.epochs),
            plot_dir=plot_dir,
            artifact_dir=str(ROOT / "results"),
        )

        preds_out = ROOT / "results/ranking_predictions.csv"
        preds.to_csv(preds_out, index=False)
        print(f"Saved: {preds_out}", flush=True)

    rankings = preds.rename(columns={"predicted_alpha": "score"})[
        ["Date", "Ticker", "score", "actual_alpha", "Period"]
    ]
    rankings = add_momentum_column(rankings)
    rankings_out = ROOT / "results/monthly_rankings.csv"
    rankings.to_csv(rankings_out, index=False)
    print(f"Saved: {rankings_out}", flush=True)

    print("Running daily-price backtest with risk rules...", flush=True)
    bt_step = int(args.rebalance_step) if args.step_size is None else int(args.step_size)
    bt = run_backtest_daily(
        predictions_df=preds[preds["Period"] == "Test"][["Date", "Ticker", "predicted_alpha", "actual_alpha"]],
        nifty_features_path=str(ROOT / "data/processed/_NSEI_features.parquet"),
        raw_dir=str(ROOT / "data/raw"),
        start=test_start,
        end=test_end,
        step_size=bt_step,
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
    )
    bt_out = ROOT / "results/backtest_results.csv"
    bt.to_csv(bt_out, index=False)
    print(f"Saved: {bt_out}", flush=True)

    print("\nDone. View results in Streamlit:", flush=True)
    print("  streamlit run dashboard/app.py", flush=True)


if __name__ == "__main__":
    main()
