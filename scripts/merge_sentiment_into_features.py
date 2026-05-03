from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from features.sentiment_integration import SentimentMergeConfig, merge_sentiment_features


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _feature_files(processed_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in processed_dir.glob("*_features.parquet")
        if p.name != "_NSEI_features.parquet"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge daily sentiment into processed feature parquet files.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hybrid_config.yaml",
        help="Config path relative to repo root.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original feature files. Default writes to *_hybrid_features.parquet",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    paths = cfg.get("paths", {})
    integration = cfg.get("feature_integration", {})

    sentiment_path = Path(paths.get("sentiment_output", "data/processed/sentiment_features.parquet"))
    processed_dir = Path(paths.get("processed_features_dir", "data/processed"))

    sentiment = pd.read_parquet(sentiment_path)
    files = _feature_files(processed_dir)
    if not files:
        raise SystemExit(f"No *_features.parquet files found in {processed_dir}")

    merge_cfg = SentimentMergeConfig(
        locf_max_days=int(integration.get("locf_max_days", 3)),
        sentiment_momentum_lookback=int(integration.get("sentiment_momentum_lookback", 5)),
        fill_missing_with_neutral=bool(integration.get("fill_missing_with_neutral", True)),
    )

    total_rows = 0
    for fp in files:
        df = pd.read_parquet(fp)
        out = merge_sentiment_features(df, sentiment, config=merge_cfg)
        if args.inplace:
            out_path = fp
        else:
            out_path = fp.with_name(fp.name.replace("_features.parquet", "_hybrid_features.parquet"))
        out.to_parquet(out_path, index=False)
        total_rows += len(out)

    print(
        f"Merged sentiment into {len(files)} files ({total_rows:,} rows). "
        f"Output mode: {'inplace' if args.inplace else 'side-by-side'}"
    )


if __name__ == "__main__":
    main()
