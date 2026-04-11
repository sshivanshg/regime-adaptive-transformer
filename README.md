# Regime-Adaptive Multimodal Transformer (RAMT)
> Short-term equity return forecasting with regime-aware deep learning

**Concepts & glossary (living notes):** [RAMT_CONCEPTS_README.md](RAMT_CONCEPTS_README.md)

## Overview
This project forecasts short-horizon equity returns using daily OHLCV data from **US** (JPM) and **Indian** NSE listings (Reliance, TCS, HDFC Bank). It combines **HMM-based regime detection** on returns and volatility with a **mixture-of-experts (MoE)** transformer architecture and **transfer learning** across markets—so predictions adapt when volatility clusters and correlations shift, rather than assuming a single static data-generating process.

## Project Structure
```
regime-adaptive-transformer/
├── data/
│   ├── download.py          ← downloads OHLCV for all tickers (2010–2026)
│   ├── raw/                 ← raw CSVs from yfinance
│   └── processed/           ← feature-engineered CSVs (36 features)
├── eda/
│   ├── eda.ipynb            ← EDA notebook (distributions, volatility, regimes)
│   └── plots/               ← generated figures for the report
├── features/
│   └── feature_engineering.py  ← 7 feature groups including HMM regimes
├── models/
│   ├── baseline_xgboost.py  ← XGBoost walk-forward baseline
│   ├── baseline_lstm.py     ← LSTM baseline (in progress)
│   └── ramt/          ← Phase 2 (planned)
├── results/
│   └── xgboost_predictions.csv  ← out-of-sample predictions
├── evaluate.py              ← metrics and results table
└── requirements.txt
```

## Tickers
| Ticker | Market | Exchange |
|--------|--------|----------|
| JPM | US Financial | NYSE |
| RELIANCE.NS | Indian Energy/Conglomerate | NSE |
| TCS.NS | Indian IT Services | NSE |
| HDFCBANK.NS | Indian Banking | NSE |
| EPIGRAL.NS | Indian Small-Cap Chemicals | NSE |

## Quickstart
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data
```bash
python data/download.py
```

### 3. Engineer features
```bash
python features/feature_engineering.py
```

### 4. Run EDA
```bash
jupyter notebook eda/eda.ipynb
```

### 5. Run baseline model
```bash
python models/baseline_xgboost.py
```

## Baseline Results (XGBoost, walk-forward validation)
| Ticker | RMSE | MAE | DA% | Sharpe |
|--------|------|-----|-----|--------|
| JPM | 0.0194 | 0.0127 | 52.13 | 0.52 |
| RELIANCE.NS | 0.0180 | 0.0121 | 52.25 | 0.54 |
| TCS.NS | 0.0151 | 0.0106 | 53.44 | 0.82 |
| HDFCBANK.NS | 0.0165 | 0.0111 | 51.52 | 0.04 |
| EPIGRAL.NS | 0.0243 | 0.0166 | 51.32 | -0.56 |
| **Average** | **0.0187** | **0.0126** | **52.13** | **0.27** |

## Feature Groups
| Group | Features | Purpose |
|-------|----------|---------|
| Lagged Returns | Return_Lag_1/2/3/5/10/20 | Price memory |
| Volatility | Realized_Vol_5/20/60, Garman_Klass, Vol_Ratio | Risk regime |
| Technical | RSI_14, MACD, Bollinger Bands | Momentum signals |
| Momentum | Momentum_5/20/60, ROC_10 | Trend direction |
| Volume | Volume_MA_Ratio, Volume_Log | Participation |
| HMM Regimes | HMM_Regime, HMM_Regime_Label | Market state |
| Cross-Asset | Rolling_Corr_Index | Market context |

## Team
| Name | Role |
|------|------|
| Vivek (230119) | Literature review, LaTeX report, theory |
| Shivansh Gupta (230054) | Data pipeline, feature engineering, models |

## Institution
B.Tech Computer Science and Artificial Intelligence
Rishihood University, Sonipat, India
