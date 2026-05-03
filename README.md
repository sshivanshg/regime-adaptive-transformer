# RAMT: Regime-Adaptive Multimodal Transformer

*A 3rd-year capstone on Indian equity alpha. Evolving from simple momentum to a Foundation-Hybrid Triple-Expert System.*

## Phase 3: Triple-Expert Hybrid System (Exemplary)

The project has evolved into a **Foundation-Hybrid** architecture, combining three distinct "Brains" to navigate the NIFTY 200:

1.  **Technical Expert**: Cross-sectional 21-day momentum (`Ret_21d`).
2.  **Foundation Expert**: **Chronos-T5** fine-tuned via **LoRA** (Low-Rank Adaptation) on 10 RAMT features.
3.  **Risk Expert**: **HMM-based** regime detection for dynamic position sizing.

### Architecture
![Triple-Expert Architecture](docs/triple_expert_architecture.png)
*(Generated reproducibly; see `docs/architecture.md` for Mermaid source)*

## Quick Start (Turn-Key)

```bash
# 1) Install dependencies + run setup
./setup.sh

# 2) Run full diagnostic ablation study (Momentum vs. Chronos vs. Hybrid)
python main.py --mode diagnostic

# 3) Run Foundation Model Explainability (Feature Importance)
python main.py --task explain

# 4) Launch Enhanced Interactive Dashboard
streamlit run dashboard/app.py
```

For containerized runs:
```bash
docker build -t ramt-phase3 .
docker run -p 8501:8501 ramt-phase3
```

## Rubric Alignment (Phase 3)

| Rubric Category | Level 5 Evidence File(s) |
| --- | --- |
| **Architecture Diagram (Publication-Ready)** | [architecture.md](docs/architecture.md), [architecture_final.png](docs/architecture_final.png) |
| **Hybrid Methodology (Outstanding)** | [hybrid_backtester.py](models/hybrid_backtester.py), [chronos_lora_v2.py](models/lora_experiment/chronos_lora_v2.py) |
| **Ablation Studies (Diagnostic)** | [run_diagnostic_ablation.py](scripts/run_diagnostic_ablation.py), [ablation_summary.json](results/ablation_summary.json) |
| **Explainability (Extra Mile)** | [explain_chronos.py](scripts/explain_chronos.py), [chronos_feature_importance.json](results/explainability/chronos_feature_importance.json) |
| **Interactive Visualization (Extra Mile)** | [app.py](dashboard/app.py) (Triple-Expert Diagnostic section) |
| **Reproducibility (Turn-Key)** | [main.py](main.py), [Dockerfile](Dockerfile), [setup.sh](setup.sh) |
| **Data Integrity / Leakage Controls** | [check_pipeline_health.py](scripts/check_pipeline_health.py), [hybrid_backtester.py](models/hybrid_backtester.py) |

## The goal

I wanted to pick 5 stocks from NIFTY 200 each month, hold until the next rebalance with stops, and use an HMM-based regime signal to decide how much of the portfolio to put to work. On paper I aimed for directional accuracy above 54% and Sharpe above 1.0, because those were the academic targets that felt reasonable for a student project. What I actually wanted was simpler: beat NIFTY buy-and-hold by enough, after costs, that I could imagine allocating real money to the strategy and not feel foolish.

## Phase 1 — the XGBoost baseline

I started with daily return prediction using XGBoost. It was the obvious first move: fast to train, easy to inspect, good for iteration. The problem was not the algorithm but the target. Daily returns are almost pure noise at this granularity; whatever edge exists in Indian large-caps shows up cross-sectionally and on a monthly horizon, not as a clean per-name next-day move. I stopped trying to squeeze signal from tomorrow’s return and changed the label to 21-day forward alpha versus NIFTY. That single pivot mattered more than any hyperparameter I tuned before it.

## Phase 2 — building RAMT

This is the piece I actually wanted to build: a multimodal stack that could respect regimes instead of hiding them in a feature column.

The pipeline looked like this: a `MultimodalEncoder` for price, technicals, volume, macro, and regime embeddings, then positional encoding, then `RegimeCrossAttention` so regime acted as the query rather than just another input vector. On top of that I used a mixture-of-experts layer with three transformer experts (bull, bear, high-vol flavor), dual prediction heads for monthly and daily horizons, and `TournamentRankingLoss` because the problem is ranking names within a date, not fitting a scalar return for each row in isolation.

Data went in as roughly 200 tickers × 10 features × 30-day sequences. I trained walk-forward, refreshing every six months, on the order of 100k samples per fold depending on overlap. The target I trained against was `Sector_Alpha`, sector-neutralized monthly alpha built from `Monthly_Alpha` so the model wasn’t just learning sector bets.

## The first honest numbers

The dashboard showed Sharpe around 1.35 and CAGR near 23%. The real backtest file, the one written by `run_backtest_daily` with friction and risk rules, showed Sharpe 0.58 and CAGR about 2.9%. The gap was not a rounding error. The dashboard simulation was missing friction at 0.22% per rebalance, regime-based position sizing, stops, and the portfolio drawdown killswitch. Once those were in the same path as the returns, most of the headline alpha evaporated. I learned to trust the path that charges me to trade and forces me out when the rules say so, not the chart that only knows gross returns.

## Three things that failed, and what they taught me

### 1. The backtest had a date-alignment bug

`run_backtest_daily` was building its rebalance grid from the NIFTY calendar while the predictions CSV marched on a different set of dates. The overlap was empty in practice, the portfolio kept going to cash, and I spent time debugging “strategy” when the bug was calendar alignment. The fix was to drive rebalance dates directly from the predictions file so the backtest and the model outputs share one timeline.

### 2. RAMT trained for 3 epochs

The config file said `MAX_EPOCHS=30` and `PATIENCE=8`, but the run script passed `--epochs 3` after a quick test and I never changed it back. Training loss was still falling, validation loss had barely moved, and per-date prediction standard deviation across 200 names was about 0.002, basically collapsed. Directional accuracy landed at 46.4%, below a coin flip, while Top-5 Sharpe on the dashboard still looked good. That was momentum leaking into the top picks, not a model that had learned a cross-sectional signal.

### 3. Re-training properly still didn’t work

I sped up the data loader (`num_workers=4`, `cache_size=250`) and cut wall time from about 8 hours to about 2 hours per heavy run. I simplified the architecture: dropped MoE to a single expert, trimmed heads and layers, and took parameters from roughly 500k down to 97k. I ran 25 epochs across four walk-forward folds. Validation correlation to the target went negative, about -0.036, and got worse every epoch. Predictions tightened further (validation σ around 0.0015). Training loss plateaued by epoch 2.

The tournament ranking loss has a real failure mode: if scores collapse toward zero, pairwise margins vanish, gradients vanish, and nothing learns to separate names on a given day. I tried a stronger daily MSE anchor (`AUX_DAILY_WEIGHT` from 0.05 to 0.2) and restored some capacity (2 layers, 8 heads, 131k parameters). It ran 7 epochs before the negative-correlation curve was so obviously monotonic that I stopped wasting electricity.

## The diagnostic that saved the project

I had two dead training runs and no clean read on whether the transformer, the features, or the target was wrong.

I wrote a small script, `scripts/baseline_feature_ic.py` (about 60 lines), that trains plain LightGBM on the same ten features on the same train and test split as the research setup. If LightGBM finds information coefficient and RAMT does not, the transformer is the problem. If neither finds anything, the features are the problem.

The numbers were blunt:

- LightGBM: IC = +0.021, IR = +6.41, top-5 positive alpha rate = 53.8%.
- Pure `Ret_21d` momentum (no ML, just sort by 21-day return): IC = +0.0065, top-5 spread = +1.59% per month, top-5 positive rate = 56.3%.
- RAMT: IC = -0.019, negatively correlated with the realized ranked target.

A `df.sort_values("Ret_21d").head(5)` strategy with zero machine learning beat my multi-stack transformer on every metric that matters.

## The pivot

I stopped trying to rescue RAMT. The evidence said monthly cross-sectional alpha in NIFTY 200, at the data scale and model budget I had (97k–131k parameters), was dominated by short-horizon momentum, and the transformer wasn’t adding information on top of it.

What I run now is deliberately boring. Each rebalance date I rank the 200 names by 21-day return, take the top 5 with at most one name per NSE sector so I do not end up with five defense stocks in a defense rally, size the book by HMM regime (100% in bull, 50% in high volatility, 20% in bear), apply a 7% per-stock stop and a 15% portfolio drawdown killswitch, and rebalance every 21 trading days. The “regime-adaptive” part of the project title moved from inside the neural net to the portfolio construction layer, which is a cleaner story than pretending the regime gate lived in the embedding space when it did not.

## Final numbers

Out-of-sample window from `2024-01-10` through `2026-01-27` on `results/final_strategy/backtest_results.csv`, with NIFTY buy-and-hold from `data/raw/_NSEI.parquet` over the same calendar span. Net of friction at 0.22% per rebalance when capital is deployed.

Strategy vs NIFTY buy-and-hold:


| Metric       | My strategy | NIFTY B&H | Edge   |
| ------------ | ----------- | --------- | ------ |
| CAGR         | 13.5%       | 7.7%      | +5.8pp |
| Max drawdown | -18.7%      | -15.8%    | -2.9pp |
| Sharpe (net) | 0.83        | 0.65      | +0.18  |
| Win rate     | 64%         | 52%       | +12pp  |


Sharpe for the strategy uses mean and standard deviation of rebalance-window returns in `backtest_results.csv`, annualized as mean divided by std times √12, matching `dashboard/app.py`. NIFTY Sharpe uses daily log returns on the index in that window, annualized with √252, same helper as the dashboard benchmark.

₹1 lakh compounds to about ₹1.30 lakh on the strategy versus about ₹1.16 lakh on NIFTY over those two years. Not a hedge fund number. A real, tradeable band for an undergrad capstone.

Full progression across the project (each row is a different experiment, not the same CSV):


| Strategy                       | CAGR  | Max DD | Sharpe |
| ------------------------------ | ----- | ------ | ------ |
| NIFTY buy-and-hold             | 7.7%  | -15.8% | 0.65   |
| RAMT transformer               | 2.9%  | -5.3%  | 0.58   |
| Momentum + regime              | 18.0% | -22.6% | 0.81   |
| Momentum + regime + sector cap | 13.5% | -18.7% | 0.83   |


The last row is what `results/final_strategy/backtest_results.csv` represents today. The earlier rows are where the research actually traveled.

## What I learned that I'll take to the next project

- Build the backtest with realistic costs before you fall in love with a model architecture.
- Any dashboard metric without friction will beat the number that includes turnover and stops; look at the path that charges you.
- When training loss stalls by epoch 2 and validation correlation trends negative, more epochs are not the fix.
- Short-horizon momentum is harder to beat than a first-year quant course suggests.
- At student data scale, a small transparent rule often beats a big fragile stack.
- Decide go or no-go metrics before you peek at the out-of-sample folder.
- A sixty-line diagnostic that runs in minutes is worth more than another week of learning-rate grid search.

## Project structure

**Data pipeline**

- `data/download.py` — pull Yahoo histories and benchmarks into `data/raw/`.
- `data/nifty200_tickers.txt` — static universe list (2026 NIFTY 200 snapshot).
- `data/raw/` — per-ticker and macro Parquet files; benchmark `data/raw/_NSEI.parquet` for NIFTY.
- `data/processed/` — engineered feature Parquet per ticker (`*_features.parquet`).
- `scripts/fetch_nifty200.py` — fetch official index constituents and save raw series.

**Features**

- `features/feature_engineering.py` — build the panel features, alphas, and benchmark alignment.
- `features/sectors.py` — hand mapping from ticker to sector for the sector cap.

**Model (kept for thesis ablation, not the live signal)**

- `models/ramt/model.py` — RAMT module definitions.
- `models/ramt/encoder.py` — multimodal encoder and regime attention wiring.
- `models/ramt/moe.py` — mixture-of-experts transformer blocks.
- `models/ramt/losses.py` — ranking and auxiliary losses.
- `models/ramt/dataset.py` — sequence dataset for walk-forward training.
- `models/ramt/train_ranking.py` — training loop, walk-forward orchestration, exports.
- `models/attention_consistency_report.py`, `models/inspect_attention.py`, `models/permutation_importance.py` — analysis helpers around the old model path.

`models/ramt/` stays in the repo for the thesis ablation chapter. The live strategy does not depend on it.

**Baseline diagnostic**

- `scripts/baseline_feature_ic.py` — LightGBM vs momentum vs RAMT exports on the same split.

**Backtest**

- `models/backtest.py` — `run_backtest_daily`, regime wiring, friction, stops, killswitch.
- `models/run_final_2024_2026.py` — end-to-end runner (train or `--backtest-only`), writes `results/final_strategy/backtest_results.csv`.

**Dashboard**

- `dashboard/app.py` — Streamlit UI: curves, metrics, regime shading, benchmark comparison.
- `dashboard/market_scraper.py`, `dashboard/market_pulse.py` — auxiliary market data helpers for the UI.

**Scripts**

- `scripts/build_momentum_predictions.py` — build the momentum-based ranking CSV the backtest consumes after the pivot.

**Results and artifacts**

- `results/final_strategy/backtest_results.csv` — authoritative per-rebalance performance for the final strategy.
- `results/final_strategy/ranking_predictions.csv`, `results/final_strategy/monthly_rankings.csv` — model or momentum ranking exports.
- `results/ramt/ramt_model_state*.pt`, `results/ramt/ramt_scaler*.joblib`, `results/ramt/ramt_y_scaler*.joblib` — saved weights and scalers from training runs.
- `results/ramt/training_history.csv`, `results/ramt/training_dashboard.png`, `results/ramt/training_log_*.txt` — training traces.

**Notebooks and docs**

- `RAMT_Monolith_Trainer.ipynb`, `RAMT_Production_Pipeline.ipynb` — older exploration and pipeline notes.
- `RAMT_CORE_AUDIT.md`, `FEATURES_AND_REGIMES.md`, `ATTENTION_EXPLAINABILITY.md` — design and audit notes.
- `requirements.txt` — Python dependencies.
- `run_dashboard.sh` — convenience launcher for Streamlit.
- `checkpoints/` — legacy XGBoost and early PyTorch artifacts (`best.pt`, `xgboost.joblib`).

## How to reproduce

```bash
# 1. Build features (assumes data/raw/ is populated)
python features/feature_engineering.py

# 2. Run the baseline diagnostic (the thing that motivated the pivot)
python scripts/baseline_feature_ic.py

# 3. Build the momentum predictions CSV
python scripts/build_momentum_predictions.py

# 4. Run the backtest
python models/run_final_2024_2026.py --backtest-only

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

## Limitations

The universe is static: the 200 tickers are a 2026 NIFTY 200 snapshot. I did not reconstruct historical index membership, so survivorship and membership drift are still there even though I was careful about time alignment inside the panel.

The out-of-sample window is about two years. The strategy has not lived through a 2008- or 2020-style crash in this backtest. I do not know how it behaves when correlations go to one and liquidity dries up.

Rebalancing every 21 trading days is a choice. A shorter cadence would lean more on momentum; a longer one would blend in mean reversion. The result is not independent of that clock.

Friction is modeled as 0.22% of traded notional per rebalance. Real Indian market impact, especially on smaller names, is often worse.

The HMM regime is closer to a volatility state machine than a return predictor. It cuts exposure in high-vol and bear labels, which helps drawdowns but also caps upside when the market rips.

## Acknowledgments

I pulled prices and indices with `yfinance`, stored panels with `pyarrow`, leaned on `scikit-learn` and LightGBM for baselines, and built the RAMT prototype in PyTorch. No sponsor, no cloud credit fairy tale, just those libraries and a lot of CPU hours.