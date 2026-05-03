"""
User Story:
Run a single command that evaluates all required ablation scenarios and exports a
grader-ready report with consistent strategy metrics and sentiment alpha diagnostics.

Implementation Approach:
Generate both vanilla and LoRA sentiment files, validate temporal alignment,
execute each toggle configuration through the ablation engine, and serialize
scenario-level predictions, backtests, and summary CSV artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ablation_engine import (
    AblationConfig,
    compute_metrics,
    run_ablation_backtest,
    sentiment_alpha_by_regime,
)
from scripts.pipeline_sentiment import run_pipeline  # type: ignore
from scripts.validate_data_alignment import validate_alignment  # type: ignore


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ablation_study")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    ap = argparse.ArgumentParser(description="Run automated ablation study and generate rubric metrics.")
    ap.add_argument("--config", type=str, default="config/hybrid_config.yaml")
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--end", type=str, default="2026-04-16")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    paths = cfg.get("paths", {})
    ab_cfg = cfg.get("ablation", {})

    logger = _setup_logger(Path(paths.get("pipeline_log", "logs/pipeline.log")))

    processed_dir = Path(paths.get("processed_features_dir", "data/processed"))
    nifty_features = Path(paths.get("nifty_features", "data/processed/_NSEI_features.parquet"))
    raw_dir = Path(paths.get("raw_dir", "data/raw"))

    out_dir = Path(paths.get("ablation_results_dir", "results/ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Checking for sentiment data...")
    has_text = Path(paths.get("text_input", "data/raw/news_text.parquet")).exists()
    
    vanilla_sent = None
    lora_sent = None
    
    if has_text:
        logger.info("Generating sentiment files for vanilla and LoRA modes...")
        vanilla_sent = run_pipeline(Path(args.config), use_lora_adapters=False)
        lora_sent = run_pipeline(Path(args.config), use_lora_adapters=True)

        logger.info("Validating data alignment before backtests...")
        ok_v, rep_v = validate_alignment(vanilla_sent, processed_dir, nifty_features, args.start, args.end)
        ok_l, rep_l = validate_alignment(lora_sent, processed_dir, nifty_features, args.start, args.end)
        if not ok_v:
            raise SystemExit(f"Vanilla sentiment alignment failed: {rep_v}")
        if not ok_l:
            raise SystemExit(f"LoRA sentiment alignment failed: {rep_l}")
    else:
        logger.warning("Missing 'data/raw/news_text.parquet'. Sentiment-based scenarios will be skipped.")

    scenarios = [
        AblationConfig(
            name="Baseline: Momentum Only",
            use_momentum=True,
            use_hmm_regime=False,
            use_finbert_sentiment=False,
            use_lora_adapters=False,
            top_n=int(ab_cfg.get("top_n", 5)),
            capital=float(ab_cfg.get("capital", 100000.0)),
        ),
        AblationConfig(
            name="ML-Enhanced: Momentum + HMM",
            use_momentum=True,
            use_hmm_regime=True,
            use_finbert_sentiment=False,
            use_lora_adapters=False,
            top_n=int(ab_cfg.get("top_n", 5)),
            capital=float(ab_cfg.get("capital", 100000.0)),
        ),
    ]

    if has_text:
        scenarios.extend([
            AblationConfig(
                name="DL-Enhanced: Momentum + FinBERT (Vanilla)",
                use_momentum=True,
                use_hmm_regime=False,
                use_finbert_sentiment=True,
                use_lora_adapters=False,
                top_n=int(ab_cfg.get("top_n", 5)),
                capital=float(ab_cfg.get("capital", 100000.0)),
            ),
            AblationConfig(
                name="Full Hybrid (Proposed): Momentum + HMM + FinBERT (LoRA)",
                use_momentum=True,
                use_hmm_regime=True,
                use_finbert_sentiment=True,
                use_lora_adapters=True,
                top_n=int(ab_cfg.get("top_n", 5)),
                capital=float(ab_cfg.get("capital", 100000.0)),
            ),
        ])

    rows: list[dict[str, object]] = []
    bt_outputs: dict[str, pd.DataFrame] = {}

    for sc in scenarios:
        logger.info("Running scenario: %s", sc.name)
        sent_path = lora_sent if sc.use_lora_adapters else vanilla_sent
        if not sc.use_finbert_sentiment:
            sent_path = None

        preds, bt = run_ablation_backtest(
            sc,
            processed_dir=processed_dir,
            nifty_features_path=nifty_features,
            raw_dir=raw_dir,
            start=args.start,
            end=args.end,
            sentiment_path=sent_path,
        )

        pred_name = sc.name.replace(" ", "_").replace(":", "").replace("+", "plus").replace("(", "").replace(")", "")
        pred_path = out_dir / f"predictions_{pred_name}.csv"
        bt_path = out_dir / f"backtest_{pred_name}.csv"
        preds.to_csv(pred_path, index=False)
        bt.to_csv(bt_path, index=False)

        bt_outputs[sc.name] = bt
        m = compute_metrics(bt, capital=sc.capital)
        rows.append(
            {
                "Scenario": sc.name,
                "use_momentum": sc.use_momentum,
                "use_hmm_regime": sc.use_hmm_regime,
                "use_finbert_sentiment": sc.use_finbert_sentiment,
                "use_lora_adapters": sc.use_lora_adapters,
                "CAGR": m["CAGR"],
                "Sharpe_Net": m["Sharpe_Net"],
                "Max_Drawdown": m["Max_Drawdown"],
                "Win_Rate": m["Win_Rate"],
                "predictions_csv": str(pred_path),
                "backtest_csv": str(bt_path),
            }
        )

    report = pd.DataFrame(rows)
    report_path = Path(paths.get("ablation_report", "results/ablation_report.csv"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    logger.info("Saved ablation report: %s", report_path)

    # Sentiment alpha diagnostics by regime for the full hybrid
    if has_text:
        full_name = "Full Hybrid (Proposed): Momentum + HMM + FinBERT (LoRA)"
        no_sent_name = "ML-Enhanced: Momentum + HMM"
        by_regime = sentiment_alpha_by_regime(bt_outputs[full_name], bt_outputs[no_sent_name])
        regime_path = out_dir / "ablation_sentiment_alpha_by_regime.csv"
        by_regime.to_csv(regime_path, index=False)
        logger.info("Saved sentiment alpha by regime: %s", regime_path)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
