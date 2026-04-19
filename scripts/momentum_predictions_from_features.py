"""
Build a ``ranking_predictions``-style CSV from processed features using **21d momentum**
(``Ret_21d``) as ``predicted_alpha`` and ``Sector_Alpha`` as ``actual_alpha``.

Use when no RAMT walk-forward exists for the window (e.g. historical yfinance download).

Example::

  python scripts/momentum_predictions_from_features.py \\
    --processed-dir data/processed_yf_2008_2010 \\
    --output results/yf_momentum_rankings_2008_2010.csv \\
    --start 2008-01-01 \\
    --end 2010-12-31 \\
    --rebalance-every 21
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument(
        "--rebalance-every",
        type=int,
        default=21,
        help="Every K-th **sorted trading day** in NIFTY calendar (default 21 ≈ monthly)",
    )
    args = ap.parse_args()

    processed = args.processed_dir.resolve()
    nifty_path = processed / "_NSEI_features.parquet"
    if not nifty_path.is_file():
        raise SystemExit(f"Missing NIFTY features: {nifty_path}")

    paths = sorted(
        p
        for p in processed.glob("*_features.parquet")
        if p.name != "_NSEI_features.parquet"
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

    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)

    n = pd.read_parquet(nifty_path, columns=["Date"])
    n["Date"] = pd.to_datetime(n["Date"])
    nd = n["Date"].sort_values().unique()
    nd = nd[(nd >= start_ts) & (nd <= end_ts)]
    if len(nd) == 0:
        raise SystemExit("No NIFTY trading days in requested window.")
    rebal = nd[:: int(args.rebalance_every)]
    rebal_set = set(pd.Timestamp(x) for x in rebal)

    panel = panel[panel["Date"].isin(rebal_set)]
    panel = panel.rename(
        columns={"Ret_21d": "predicted_alpha", "Sector_Alpha": "actual_alpha"}
    )
    panel["actual_alpha"] = pd.to_numeric(panel["actual_alpha"], errors="coerce").fillna(0.0)
    panel["predicted_alpha"] = pd.to_numeric(panel["predicted_alpha"], errors="coerce").fillna(
        0.0
    )
    panel["Period"] = "Test"

    out = panel[["Date", "Ticker", "predicted_alpha", "actual_alpha", "Period"]].sort_values(
        ["Date", "Ticker"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows → {args.output}")


if __name__ == "__main__":
    main()
