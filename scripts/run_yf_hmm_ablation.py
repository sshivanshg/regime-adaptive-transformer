"""
End-to-end: yfinance download → features → momentum rankings → HMM vs flat backtest.

Does not touch ``data/raw`` or ``data/processed`` defaults; uses parallel trees under ``data/``.

Examples::

  # 2008–2010 (default-style: full NIFTY 200 fetch from NSE CSV)
  python scripts/run_yf_hmm_ablation.py \\
    --start 2008-01-01 --end-exclusive 2011-01-01 \\
    --bt-start 2008-01-01 --bt-end 2010-12-31 --tag 2008_2010

  # 2010–2012 / 2013–2015 with a period-agnostic large-cap proxy list (see scripts/universe/)
  python scripts/run_yf_hmm_ablation.py \\
    --start 2010-01-01 --end-exclusive 2013-01-01 \\
    --bt-start 2010-01-01 --bt-end 2012-12-31 --tag 2010_2012 \\
    --universe-file scripts/universe/nifty100_nse_survivorship_proxy.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="YFinance pipeline + HMM vs flat ablation")
    ap.add_argument("--start", default="2008-01-01", help="yfinance download start (inclusive)")
    ap.add_argument(
        "--end-exclusive",
        default="2011-01-01",
        help="yfinance download end (exclusive); use 2011-01-01 to include through 2010-12-31",
    )
    ap.add_argument("--bt-start", default="2008-01-01", help="Backtest window start")
    ap.add_argument("--bt-end", default="2010-12-31", help="Backtest window end (inclusive)")
    ap.add_argument(
        "--tag",
        default="2008_2010",
        help="Suffix for data/raw_yf_<tag>, data/processed_yf_<tag>, results/momentum_rankings_yf_<tag>.csv",
    )
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-momentum", action="store_true")
    ap.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="Optional: pass through to fetch_nifty200.py (one Yahoo ticker per line)",
    )
    args = ap.parse_args()

    raw_dir = ROOT / "data" / f"raw_yf_{args.tag}"
    proc_dir = ROOT / "data" / f"processed_yf_{args.tag}"
    pred_csv = ROOT / "results" / f"momentum_rankings_yf_{args.tag}.csv"
    py = sys.executable

    if not args.skip_download:
        cmd = [
            py,
            str(ROOT / "scripts" / "fetch_nifty200.py"),
            "--start",
            args.start,
            "--end",
            args.end_exclusive,
            "--raw-dir",
            str(raw_dir),
        ]
        if args.universe_file is not None:
            cmd.extend(["--universe-file", str(Path(args.universe_file).resolve())])
        run(cmd)

    if not args.skip_features:
        run(
            [
                py,
                str(ROOT / "scripts" / "build_processed_range.py"),
                "--raw-dir",
                str(raw_dir),
                "--processed-dir",
                str(proc_dir),
                "--start",
                args.start,
                "--end",
                args.end_exclusive,
            ]
        )

    if not args.skip_momentum:
        run(
            [
                py,
                str(ROOT / "scripts" / "momentum_predictions_from_features.py"),
                "--processed-dir",
                str(proc_dir),
                "--output",
                str(pred_csv),
                "--start",
                args.bt_start,
                "--end",
                args.bt_end,
            ]
        )

    out_hmm = ROOT / "results" / "hmm_vs_flat" / f"yf_{args.tag}"
    run(
        [
            py,
            str(ROOT / "scripts" / "hmm_vs_flat_backtest.py"),
            "--start",
            args.bt_start,
            "--end",
            args.bt_end,
            "--predictions",
            str(pred_csv),
            "--raw-dir",
            str(raw_dir),
            "--nifty-features",
            str(proc_dir / "_NSEI_features.parquet"),
            "--out-dir",
            str(out_hmm),
        ]
    )

    print("\nDone. Look for hmm_vs_flat_summary.csv under:", out_hmm, flush=True)


if __name__ == "__main__":
    main()
