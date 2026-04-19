# RAMT - Regime-Adaptive Monthly Transformer

RAMT is a research and simulation pipeline for ranking NIFTY stocks by next-month relative performance and running regime-aware portfolio backtests.

The project is now in a near-production research state with:

- NIFTY 200 Parquet data pipeline
- sector-neutral monthly alpha target
- walk-forward retraining with strict train/test separation
- regime-aware portfolio sizing
- Streamlit dashboard with live simulation and market pulse scraper

## Current Status (as of 2026-04-19)

Implemented and working in this repository:

- End-to-end flow: ingest -> features -> training -> ranking outputs -> backtest outputs
- Causal feature engineering over NIFTY 200 style universe
- Monthly target: `Sector_Alpha` (fallback to `Monthly_Alpha`)
- Dual-head model objective (monthly ranking + daily return auxiliary head)
- Tournament/ranking-focused training objective and walk-forward folds
- Regime conditioning with `HMM_Regime` + regime-aware allocation rules
- Artifact export for model/scalers and dashboard consumption
- Interactive dashboard for live simulation, analytics, and raw data inspection

## Quick Start

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended Pipeline (Current)

1) Fetch NIFTY 200 + macro raw data to Parquet

```bash
source .venv/bin/activate
python scripts/fetch_nifty200.py
```

2) Build processed features and targets

```bash
python -m features.feature_engineering
```

3) Train + walk-forward inference + backtest

```bash
python -m models.run_final_2024_2026 --epochs 3
```

4) Launch dashboard

```bash
./run_dashboard.sh
```

Dashboard opens at http://localhost:8501.

## Outputs You Should Expect

After a successful run, the `results/` directory is refreshed with core artifacts:

- `ranking_predictions.csv`
- `monthly_rankings.csv`
- `backtest_results.csv`
- `training_history.csv`
- `training_dashboard.png`
- `ramt_model_state.pt`
- `ramt_scaler.joblib`
- `ramt_y_scaler.joblib`
- walk-forward segment checkpoints/scalers (for each segment)

## What the Model Predicts

Strategic monthly objective:

- `Monthly_Alpha` = stock forward 21d return minus NIFTY forward 21d return
- `Sector_Alpha` = `Monthly_Alpha` demeaned by same-date same-sector median

Training target currently prioritizes:

- `Sector_Alpha` when present
- fallback to `Monthly_Alpha` when sector-neutral column is absent

Auxiliary tactical objective:

- `Daily_Return` (next-day return) as a secondary head for stability/sanity

## Features and Regime Inputs

Current RAMT feature vector uses a lean schema defined in `models/ramt/dataset.py`:

- Price returns: `Ret_1d`, `Ret_5d`, `Ret_21d`
- Technicals: `RSI_14`, `BB_Dist`
- Volume: `Volume_Surge`
- Macro lagged returns: `Macro_INDIAVIX_Ret1d_L1`, `Macro_CRUDE_Ret1d_L1`, `Macro_USDINR_Ret1d_L1`, `Macro_SP500_Ret1d_L1`

Regime handling:

- `HMM_Regime` is passed separately to the model (not mixed into scaled feature vector)

## Portfolio Logic (Backtest)

At each rebalance window (default 21 trading days):

- Rank stocks by model score
- Read NIFTY regime
- Apply risk posture by regime

Default posture:

- Bull: fully invested, broader basket
- High-vol: reduced exposure and fewer positions
- Bear: risk-off behavior (cash/defensive stance)

Backtest includes market friction assumptions and walk-forward evaluation behavior.

## Dashboard Capabilities

The Streamlit app in `dashboard/app.py` now includes:

- Live performance simulation tab
- Training analytics tab
- Raw data explorer tab
- Regime-colored visuals and shadow portfolio conviction signals
- Market pulse integration from scraper utilities in `dashboard/market_scraper.py`

## Repository Map

```text
regime-adaptive-transformer/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- download.py
|-- scripts/
|   `-- fetch_nifty200.py
|-- features/
|   |-- feature_engineering.py
|   `-- sectors.py
|-- models/
|   |-- run_final_2024_2026.py
|   |-- backtest.py
|   |-- inspect_attention.py
|   |-- attention_consistency_report.py
|   |-- permutation_importance.py
|   `-- ramt/
|       |-- dataset.py
|       |-- model.py
|       |-- encoder.py
|       |-- moe.py
|       `-- train_ranking.py
|-- dashboard/
|   |-- app.py
|   |-- market_pulse.py
|   `-- market_scraper.py
|-- RAMT_Monolith_Trainer.ipynb
|-- RAMT_Production_Pipeline.ipynb
|-- FEATURES_AND_REGIMES.md
|-- ATTENTION_EXPLAINABILITY.md
|-- requirements.txt
`-- README.md
```

## Recent Milestones Completed

- Migrated to NIFTY 200-oriented training flow
- Added NIFTY feature parity and improved causal safeguards
- Added live market scraping support for dashboard pulse
- Strengthened walk-forward logic and compounding behavior
- Added tournament ranking objective and regime cross-attention path
- Integrated sector-neutral alpha target into training workflow

## Notes and Limitations

- Research system only; not financial advice
- Universe construction can still carry survivorship bias depending on raw universe snapshot
- Regime model and execution assumptions should be stress-tested before any real deployment

## Optional Workflows

- Notebook-first training: `RAMT_Monolith_Trainer.ipynb`
- Remote/GPU production-style runbook: `RAMT_Production_Pipeline.ipynb`

## Last Updated

2026-04-19