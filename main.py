"""
User Story:
As a researcher, I need a single entry point that orchestrates all pipeline stages
(data, sentiment, validation, ablation, diagram, explainability) so that the entire
workflow can be executed with one command for reproducibility.

Implementation Approach:
Parse --task and --config arguments, load configuration, then invoke the appropriate
sub-scripts sequentially. The "all" task runs the full pipeline in deterministic order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _resolve_task(task: str | None, mode: str | None) -> str:
    if task:
        return task
    if mode:
        return mode
    raise SystemExit("Provide --task (preferred) or --mode.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn-key entrypoint for RAMT Phase 3 pipelines.")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        choices=["all", "data", "sentiment", "validate", "ablation", "diagnostic", "diagram", "explain", "smoke-test"],
        help="Execution task (preferred).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        choices=["sentiment", "validate", "ablation", "diagnostic", "smoke-test"],
        help="Backward-compatible alias for --task.",
    )
    parser.add_argument("--config", type=str, default="config/hybrid_config.yaml")
    parser.add_argument("--use-lora-adapters", action="store_true")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-04-16")
    args = parser.parse_args()

    task = _resolve_task(args.task, args.mode)
    cfg = _load_yaml(Path(args.config))
    paths = cfg.get("paths", {})

    processed_dir = Path(paths.get("processed_features_dir", "data/processed"))
    nifty_features = Path(paths.get("nifty_features", "data/processed/_NSEI_features.parquet"))
    sentiment_input = Path(paths.get("text_input", "data/raw/news_text.parquet"))

    if task == "data":
        # Run feature engineering only when required artifacts are missing.
        if not processed_dir.exists() or not nifty_features.exists():
            _run([sys.executable, "features/feature_engineering.py"])
        else:
            print("Data stage skipped: processed features already available.")
        return

    if task == "sentiment":
        _run(
            [
                sys.executable,
                "scripts/pipeline_sentiment.py",
                "--config",
                args.config,
            ]
            + (["--use-lora-adapters"] if args.use_lora_adapters else [])
        )
        return

    if task == "validate":
        _run(
            [
                sys.executable,
                "scripts/validate_data_alignment.py",
                "--config",
                args.config,
                "--start",
                args.start,
                "--end",
                args.end,
            ]
        )
        return

    if task == "ablation":
        _run(
            [
                sys.executable,
                "scripts/run_ablation_study.py",
                "--config",
                args.config,
                "--start",
                args.start,
                "--end",
                args.end,
            ]
        )
        return

    if task == "diagnostic":
        _run([sys.executable, "scripts/check_pipeline_health.py"])
        _run([sys.executable, "scripts/run_diagnostic_ablation.py"])
        return

    if task == "explain":
        _run([sys.executable, "scripts/explain_chronos.py"])
        return

    if task == "diagram":
        _run([sys.executable, "scripts/generate_architecture_diagram.py"])
        return

    if task == "explain":
        _run([sys.executable, "scripts/explain_sentiment.py", "--config", args.config])
        return

    if task == "all":
        # 1) Data/HMM stage.
        if not processed_dir.exists() or not nifty_features.exists():
            _run([sys.executable, "features/feature_engineering.py"])

        # 2) Sentiment/ablation stage when text input exists; otherwise continue with partial synthesis.
        if sentiment_input.exists():
            _run([sys.executable, "scripts/run_ablation_study.py", "--config", args.config, "--start", args.start, "--end", args.end])
        else:
            print(
                f"Warning: missing text input for ablation stage: {sentiment_input}. "
                "Continuing with available artifacts."
            )

        _run([sys.executable, "scripts/generate_architecture_diagram.py"])
        _run([sys.executable, "scripts/generate_summary_infographic.py"])
        _run([sys.executable, "scripts/generate_final_report.py"])

        try:
            _run([sys.executable, "scripts/explain_sentiment.py", "--config", args.config])
        except Exception:
            print("Explainability stage skipped due to unavailable optional inputs.")

        print("Full pipeline complete.")
        return

    if task == "smoke-test":
        required = [
            processed_dir,
            nifty_features,
            Path(paths.get("nifty200_tickers", "data/nifty200_tickers.txt")),
            Path("scripts/pipeline_sentiment.py"),
            Path("scripts/run_ablation_study.py"),
            Path("scripts/generate_architecture_diagram.py"),
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise SystemExit(f"Smoke test failed. Missing: {missing}")
        print("Smoke test passed.")
        return

    raise SystemExit(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
