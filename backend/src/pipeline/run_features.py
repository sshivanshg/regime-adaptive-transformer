from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature engineering with parameterized dates/dirs.")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None, help="Exclusive end date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Import from repo code (assumes cwd=repo root)
    from features.feature_engineering import configure_pipeline  # type: ignore
    import features.feature_engineering as fe  # type: ignore

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    configure_pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        start_date=args.start,
        end_date_exclusive=args.end,
    )

    # Run the pipeline
    fe.main()


if __name__ == "__main__":
    main()

