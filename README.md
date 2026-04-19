# RAMT - Regime-Adaptive Monthly Stock Ranking (NIFTY)

RAMT is a research-first, production-oriented monthly ranking system for Indian equities.

It combines:

- Strategic monthly ranking: predicts next-month relative outperformance vs NIFTY.
- Tactical daily head: predicts next-day return as a sanity anchor.
- Regime-aware execution: uses NIFTY HMM regime for allocation/risk posture.

This README is intentionally in-depth and updated to reflect the current codebase state.

## 1) Current Project Status (2026-04-19)

Implemented and integrated end-to-end:

- NIFTY 200-oriented raw ingestion to Parquet (`scripts/fetch_nifty200.py`).
- Causal feature engineering to processed Parquet (`features/feature_engineering.py`).
- Sector mapping and sector-neutral target generation (`Sector_Alpha`).
- Walk-forward RAMT training with train-only scaling and winsorization.
- Tournament-style cross-sectional ranking loss + auxiliary daily MSE loss.
- Regime cross-attention + MoE routing in model core.
- Daily price backtest with friction, stop-loss, and Kelly-style weighting options.
- Streamlit production dashboard with simulation, analytics, raw explorer, and market pulse scraping.

## 2) Environment Setup

Run from repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Recommended End-to-End Runbook

### Step A: Raw data ingestion (NIFTY 200 + macro)

```bash
source .venv/bin/activate
python scripts/fetch_nifty200.py
```

What it does:

- Downloads equity OHLCV + Adj Close into `data/raw/*.parquet`.
- Downloads macro series (`INDIAVIX`, `CRUDE`, `USDINR`, `SP500`) as `data/raw/macro_*.parquet`.
- Uses period `2020-01-01` to `2026-04-15` (end-exclusive API handling uses `2026-04-16`).

### Step B: Feature engineering

```bash
python -m features.feature_engineering
```

What it does:

- Builds per-ticker processed files in `data/processed/*_features.parquet`.
- Generates `Monthly_Alpha`, `Sector_Alpha`, `Daily_Return`, `HMM_Regime`, `Sector`.
- Applies panel step to compute sector-neutral target (`apply_sector_alpha_panel`).

### Step C: Train + walk-forward inference + backtest

```bash
python -m models.run_final_2024_2026 --epochs 3
```

What it does:

- Runs walk-forward RAMT training and writes ranking outputs.
- Writes model/scaler artifacts for dashboard and explainability tools.
- Runs backtest on test-period predictions and writes backtest CSV.

### Step D: Launch dashboard

```bash
./run_dashboard.sh
```

Dashboard URL:

- http://localhost:8501

## 4) Fast CLI Reference (`models/run_final_2024_2026.py`)

```bash
python -m models.run_final_2024_2026 \
  --epochs 3 \
  --training-step 126 \
  --rebalance-step 21 \
  --inference-warmup-days 30
```

Useful options:

- `--backtest-only`: skip training and reuse existing predictions CSV.
- `--predictions <path>`: custom predictions CSV for backtest-only mode.
- `--batch-size`: override RAMT training batch size.
- `--patience`: override early stopping patience.
- `--no-plots`: skip `training_history.csv` and `training_dashboard.png` generation.
- `--step-size`: deprecated alias for rebalance step (kept for backward compatibility).

## 5) Data Pipeline Details

### 5.1 Raw Data

Primary current path:

- `scripts/fetch_nifty200.py` -> Parquet outputs in `data/raw/`.

Legacy path (still present in repo):

- `data/download.py` (CSV-oriented older flow).

### 5.2 Processed Feature Schema (`features/feature_engineering.py`)

Each processed parquet follows a canonical engineered schema (`FEATURE_OUTPUT_COLUMNS`):

- `Date`, `Ticker`
- `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- `Ret_1d`, `Ret_5d`, `Ret_21d`
- `Realized_Vol_20`
- `RSI_14`
- `BB_Dist`
- `Volume_Surge`
- `Macro_INDIAVIX_Ret1d_L1`
- `Macro_CRUDE_Ret1d_L1`
- `Macro_USDINR_Ret1d_L1`
- `Macro_SP500_Ret1d_L1`
- `Monthly_Alpha`
- `Sector_Alpha`
- `Daily_Return`
- `HMM_Regime`
- `Sector`

### 5.3 Feature Engineering Logic (current)

- Returns use Adj Close log-return formulation.
- Macro returns are merged with one-day lag to reduce leakage.
- HMM regime uses expanding-window Gaussian HMM over (`Ret_1d`, `Realized_Vol_20`).
- Regime semantic mapping:
  - Bull = 1 (highest mean return state)
  - High-vol = 0 (middle state)
  - Bear = 2 (lowest mean return state)
- Warm-up NaNs are kept and filled in selected columns to avoid dropping early rows.

### 5.4 Sector Mapping

- Sector labels come from `features/sectors.py` via `get_sector()`.
- Unknown symbols fall back to `OTHER`.
- `Sector_Alpha` is computed panel-wise as:

```text
Sector_Alpha = Monthly_Alpha - median(Monthly_Alpha | Date, Sector)
```

## 6) Model Inputs and Targets

### 6.1 Model feature vector (`models/ramt/dataset.py`)

Current RAMT `ALL_FEATURE_COLS` uses 10 scaled numeric features:

- Price: `Ret_1d`, `Ret_5d`, `Ret_21d`
- Technical: `RSI_14`, `BB_Dist`
- Volume: `Volume_Surge`
- Macro: `Macro_INDIAVIX_Ret1d_L1`, `Macro_CRUDE_Ret1d_L1`, `Macro_USDINR_Ret1d_L1`, `Macro_SP500_Ret1d_L1`

Regime is handled separately:

- `HMM_Regime` is not part of the scaled feature vector `X`; it is fed as a separate conditioning signal.

### 6.2 Targets

- Primary monthly target: `Sector_Alpha`.
- Fallback monthly target: `Monthly_Alpha` if `Sector_Alpha` is missing.
- Auxiliary tactical target: `Daily_Return`.

## 7) RAMT Architecture (Current)

Core pipeline in `models/ramt/model.py`:

1. `MultimodalEncoder`
2. `PositionalEncoding`
3. `RegimeCrossAttention`
4. `MixtureOfExperts`
5. Dual heads:
   - `monthly_head`
   - `daily_head`

Model output:

- Monthly prediction
- Daily prediction
- Expert gate weights

## 8) Training Strategy and Loss Design

Training implementation: `models/ramt/train_ranking.py`.

### 8.1 Walk-forward design

- Retraining cadence (`training_step`): default 126 trading days (~6 months).
- Inference/rebalance cadence (`rebalance_step`): default 21 trading days.
- Extra scoring warm-up requirement: `SEQ_LEN + inference_warmup_days`.

### 8.2 Scaling and leakage controls

- Feature scaling: `RobustScaler` (global or sector-neutral mode).
- Monthly label winsorization bounds computed from training keys only (1st/99th percentile).
- Monthly label scaler fit on training-only winsorized labels.
- Validation uses trailing window from train period; test dates are out-of-sample.

### 8.3 Objective

Current default (`USE_TOURNAMENT_LOSS=True`):

- Strategic ranking term: full pairwise magnitude-weighted tournament loss (unscaled monthly alpha space).
- Auxiliary term: daily MSE with small weight (`AUX_DAILY_WEIGHT=0.05`).

Additional mechanics in training:

- High-vol sample upweighting (`HIGH_VOL_SAMPLE_WEIGHT=2.0` for regime 0).
- Time-decay weighting:
  - 2020 samples downweighted (0.5x)
  - 2024-2026 samples upweighted (2.0x)
- LR warmup + ReduceLROnPlateau + gradient clipping + early stopping.

## 9) Portfolio Backtest Design

Backtest implementation: `models/backtest.py` (`run_backtest_daily`).

### 9.1 Regime posture

- Bull (1): high exposure.
- High-vol (0): reduced exposure, fewer names.
- Bear (2): fractional toe-hold / defensive exposure.

### 9.2 Current defaults in final runner

From `models/run_final_2024_2026.py` call:

- `top_n=5`
- `capital=100000`
- `stop_loss=0.07`
- `max_weight=0.25`
- `portfolio_dd_cash_trigger=0.15`
- `rebalance_friction_rate=0.002` (internally normalized to real-2026 friction constant)
- `kelly_p=0.5238`
- `kelly_use_predicted_margin=True`
- `kelly_scale_position=True`

### 9.3 Backtest mechanics

- Trades on rebalance grid from NIFTY trading calendar.
- Applies geometric compounding of NAV.
- Includes rebalance friction cost (trade-value based).
- Supports Kelly-style softmax allocation and optional turnover-aware selection penalty.

## 10) Artifacts and Outputs

Primary outputs in `results/`:

- `ranking_predictions.csv`
- `monthly_rankings.csv`
- `backtest_results.csv`
- `training_history.csv`
- `training_dashboard.png`
- `ramt_model_state.pt`
- `ramt_scaler.joblib`
- `ramt_y_scaler.joblib`
- Walk-forward snapshots:
  - `ramt_model_state_wf_seg_XX.pt`
  - `ramt_scaler_wf_seg_XX.joblib`
  - `ramt_y_scaler_wf_seg_XX.joblib`

`ranking_predictions.csv` includes `Period` labels (`Train`/`Test`) for downstream filtering.

## 11) Dashboard (Streamlit)

Main app: `dashboard/app.py`.

Current tabs:

- Live performance
- Training analytics
- Raw data explorer

Integrated components:

- Ranking predictions + model artifacts visualization
- Regime-aware simulation and shadow portfolio view
- Market pulse data through `dashboard/market_scraper.py`

Runner script:

- `run_dashboard.sh`

## 12) Explainability and Feature Audit Tools

### Attention tools

- `models/inspect_attention.py`
- `models/attention_consistency_report.py`

These generate attention exports under `results/`.

Note:

- These scripts currently read processed CSV paths internally; if your workflow is Parquet-only, adapt paths or keep matching CSV exports for those utilities.

### Permutation feature importance

- `models/permutation_importance.py`

Generates:

- `results/permutation_importance.csv`

Note:

- This utility also expects legacy CSV processed paths in its current implementation.

## 13) Notebooks

- `RAMT_Monolith_Trainer.ipynb`: single-notebook training stack mirroring RAMT logic.
- `RAMT_Production_Pipeline.ipynb`: structured remote/GPU production pipeline notebook (ingest -> features -> train -> artifact zip).

## 14) Dependencies

Core packages from `requirements.txt`:

- Data: `numpy`, `pandas`, `yfinance`, `pyarrow`
- ML: `torch`, `scikit-learn`, `hmmlearn`, `xgboost`
- Dashboard: `streamlit`, `plotly`, `requests`, `beautifulsoup4`
- Research/EDA: `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `jupyter`, `ipykernel`
- Utility: `tqdm`

## 15) Repository Structure

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
|       |-- encoder.py
|       |-- moe.py
|       |-- model.py
|       |-- losses.py
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

## 16) Important Notes and Known Caveats

- Research system only. Not financial advice.
- Universe handling can still carry survivorship bias depending on ticker snapshot source.
- Some explainability/audit scripts still reference legacy CSV paths while the primary pipeline is Parquet-first.
- Strategy and friction assumptions are configurable and should be stress-tested before any live capital use.

## 17) Troubleshooting

If training errors mention missing `Sector` or `Sector_Alpha`:

1. Re-run feature engineering end-to-end:

```bash
python -m features.feature_engineering
```

2. Re-run final pipeline:

```bash
python -m models.run_final_2024_2026 --epochs 3
```

If dashboard does not show fresh rankings:

1. Confirm `results/ranking_predictions.csv` exists and has recent dates.
2. Confirm model artifacts exist:
   - `results/ramt_model_state.pt`
   - `results/ramt_scaler.joblib`
   - `results/ramt_y_scaler.joblib`

## 18) Recent Improvement Timeline

Recent integrated improvements (from current repo state and recent commits):

- Pivoted to monthly relative ranking and expanded NIFTY universe handling.
- Added stricter causal safeguards in feature + backtest flow.
- Added tournament ranking objective and regime cross-attention.
- Improved walk-forward cadence and compounding behavior.
- Added market pulse scraping integration for dashboard.
- Consolidated production artifacts for fold-wise model/scaler snapshots.

## Last Updated

2026-04-19