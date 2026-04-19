"""
Build ``data/processed/*_features.parquet`` for a custom raw directory and date window.

Must match a prior ``scripts/fetch_nifty200.py`` download (same ``--raw-dir``, ``--start``, ``--end``).

Example::

  python scripts/build_processed_range.py \\
    --raw-dir data/raw_yf_2008_2010 \\
    --processed-dir data/processed_yf_2008_2010 \\
    --start 2008-01-01 \\
    --end 2011-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from features import feature_engineering as fe

    ap = argparse.ArgumentParser(description="Feature-engineer raw Parquet dir into processed/")
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument(
        "--start",
        type=str,
        required=True,
        help="Calendar lower bound (aligned with fetch --start)",
    )
    ap.add_argument(
        "--end",
        type=str,
        required=True,
        help="Exclusive end date (same as fetch --end)",
    )
    args = ap.parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    fe.configure_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        start_date=args.start,
        end_date_exclusive=args.end,
    )

    bench_path = fe._download_benchmark_if_missing()
    nifty_df = fe._read_raw_equity(Path(bench_path))
    macro_data = fe.load_macro_series(raw_dir)
    equity_paths = fe.list_equity_input_paths(raw_dir)
    if not equity_paths:
        raise SystemExit(f"No equity inputs under {raw_dir}")

    print(f"Processing {len(equity_paths)} raw files → {processed_dir} …")
    for p in equity_paths:
        try:
            fe.process_raw_equity_path(p, nifty_df, macro_data, processed_dir)
        except Exception as e:
            print(f"Skipping {p.stem}: {type(e).__name__}: {e}")

    fe.apply_sector_alpha_panel(processed_dir)
    print("Done.")


if __name__ == "__main__":
    main()
