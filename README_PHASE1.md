# README_PHASE1: Regime-Adaptive Sentiment Hybrid

This phase upgrades RAMT from a pure momentum + regime-sleeve strategy into a logical hybrid where sentiment explicitly addresses momentum failure modes under stress.

## Why this is logically integrated

The original strategy ranked by momentum (`Ret_21d`) and only used HMM regimes for sizing. In this phase:

- Bull / low-vol regime keeps momentum as primary, because trend persistence dominates.
- High-vol regime uses an integrated score (`0.6 * momentum + 0.4 * sentiment`) to reduce whipsaw risk.
- Bear / high-vol regime switches to defensive sentiment gating and can stay in cash when conviction is low.

This creates regime-conditional decision logic, not just feature concatenation.

## Components added

1. `scripts/generate_sentiment_features.py`
- Loads FinBERT (`ProsusAI/finbert`) for text sentiment.
- Loads `ChronosLoRARanker` from `models/lora_experiment/chronos_lora.py` as a calibration layer (optional blend).
- Reads text input (`Date`, `Ticker`, text column), filters to NIFTY 200 universe.
- Writes daily sentiment parquet at `data/processed/sentiment_features.parquet` with:
  - `Date`
  - `Ticker`
  - `sentiment_score` in [-1, 1]
  - `sentiment_confidence` in [0, 1]

2. `features/sentiment_integration.py`
- Merges sentiment into feature tables on `(Date, Ticker)`.
- Applies LOCF with stale cap (`limit=3` days by default).
- Adds `sentiment_momentum` as 5-day sentiment delta.

3. `scripts/merge_sentiment_into_features.py`
- Batch-merges sentiment into all `data/processed/*_features.parquet` files.
- Can write side-by-side (`*_hybrid_features.parquet`) or overwrite in place.

4. `models/hybrid_strategy.py`
- Implements hybrid ranking logic by regime:
  - Bull: Top-N momentum with sentiment floor (`sentiment > -0.2`)
  - High-vol: weighted integrated score (`0.6 momentum + 0.4 sentiment`)
  - Bear: sentiment-primary (`sentiment > 0.5`), else cash
- Includes `check_architecture_integrity()` to validate:
  - LoRA checkpoint load path
  - T-1 alignment of regime/sentiment/momentum for T decisions
  - leakage guards for lagged inputs

## Config-driven reproducibility

Configuration lives in `config/hybrid_config.yaml`.

All paths are repo-relative, so runs are portable across machines as long as folder structure is preserved.

## Expected data contract

Input text parquet (`paths.text_input`) should include:

- `Date`
- `Ticker`
- one of `text`, `headline`, `summary`, `content`

Output sentiment parquet:

- `(Date, Ticker)` key
- `sentiment_score`
- `sentiment_confidence`

## Typical run order

```bash
python scripts/generate_sentiment_features.py --config config/hybrid_config.yaml
python scripts/merge_sentiment_into_features.py --config config/hybrid_config.yaml --inplace
```

Then build momentum predictions/backtest as before, but route rankings through `models/hybrid_strategy.py` to apply regime-specific gating.
