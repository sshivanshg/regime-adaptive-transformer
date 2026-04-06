# RAMT Phase 2 — Implementation Plan
## 15-Day Deep Learning Development Roadmap

---

## Project Context

### What We Have (Phase 1 Complete)
- Data pipeline: 5 tickers, 2010-2026, 35 features per ticker
- Feature engineering: HMM regime labels, rolling correlation, technicals
- XGBoost baseline: regime-stratified, walk-forward, DA=52.13%, Sharpe=0.27
- Processed CSVs: data/processed/{TICKER}_features.csv (zero NaNs)

### What We Must Build (Phase 2)
Per our submitted proposal, Phase 2 requires:
1. LSTM baseline (proper PyTorch implementation)
2. Full RAMT architecture:
   - Multimodal Feature Encoders
   - Regime-Aware Mixture of Experts (MoE)
   - Transformer temporal backbone
   - Cross-market transfer learning
3. Combined loss function (MSE + Directional)
4. Walk-forward training for all DL models
5. Full comparison table: XGBoost vs LSTM vs RAMT
6. Additional metrics: Max Drawdown, Profit Factor, Calmar Ratio

### Target Metrics (From Proposal)
- RMSE < 2% on next-day log returns
- Sharpe Ratio > 2.0 in walk-forward backtests
- DA% > 54% (beat XGBoost baseline of 52.13%)

---

## Architecture Overview

### Data Flow
data/processed/{TICKER}_features.csv
↓
DataLoader (sequences of 30 days)
↓
┌─────────────────────────────┐
│     RAMT Model              │
│                             │
│  MultimodalEncoder          │
│  (per feature group)        │
│         ↓                   │
│  Transformer Backbone       │
│  (temporal attention)       │
│         ↓                   │
│  MoE Gating                 │
│  (regime-conditioned)       │
│         ↓                   │
│  Return Prediction Head     │
└─────────────────────────────┘
↓
Next-day log return prediction

### Feature Groups for Encoders
Group 1 — Price/Returns (6 features):
Return_Lag_1, _2, _3, _5, _10, _20
Group 2 — Volatility (5 features):
Realized_Vol_5, _20, _60, Garman_Klass_Vol, Vol_Ratio
Group 3 — Technical (8 features):
RSI_14, MACD, MACD_Signal, MACD_Hist,
BB_Upper, BB_Lower, BB_Width, BB_Position
Group 4 — Momentum (4 features):
Momentum_5, _20, _60, ROC_10
Group 5 — Volume (2 features):
Volume_MA_Ratio, Volume_Log
Group 6 — Regime (1 feature):
HMM_Regime
Group 7 — Cross-Asset (1 feature):
Rolling_Corr_Index
Total: 27 numeric features
(HMM_Regime_Label is string — excluded from model input)

---

## File Structure To Build
models/
├── baseline_lstm.py          ← Day 1-2
├── baseline_xgboost.py       ← Already done
├── ramt/
│   ├── init.py           ← Day 3
│   ├── dataset.py            ← Day 3 (DataLoader)
│   ├── encoder.py            ← Day 4-5
│   ├── moe.py                ← Day 6-7
│   ├── model.py              ← Day 8
│   └── losses.py             ← Day 8
├── train.py                  ← Day 9-10
├── walk_forward_dl.py        ← Day 11-12
└── evaluate_all.py           ← Day 13-14

---

## Day-by-Day Plan

---

### DAY 1-2 — LSTM Baseline
**File:** models/baseline_lstm.py
**Goal:** Proper PyTorch LSTM, same walk-forward as XGBoost

**Architecture:**
- Input: sequence of 30 days × 27 features
- LSTM Layer 1: hidden_size=64, dropout=0.2
- LSTM Layer 2: hidden_size=32, dropout=0.2
- Linear output: 32 → 1 (next-day return)

**Training per fold:**
- StandardScaler fit on training window only (no leakage)
- Adam optimizer, lr=0.001
- MSE loss
- Early stopping: patience=10 epochs
- Batch size: 32
- Max epochs: 50

**Same walk-forward as XGBoost:**
- Initial train: 60% of data
- Step size: 63 trading days
- Test: 63 days per fold
- Scaler refit every fold

**Output:**
- results/lstm_predictions.csv (Date, Ticker, y_true, y_pred)
- Print same metrics table as XGBoost (RMSE, MAE, DA%, Sharpe)

**Success criteria:**
- Runs without errors on all 5 tickers
- DA% > 50% on at least 3 tickers
- Results saved to CSV

---

### DAY 3 — Dataset Module
**File:** models/ramt/dataset.py
**Goal:** Clean DataLoader for sequence data

**What it must do:**
- Load data/processed/{TICKER}_features.csv
- Create sequences of length seq_len=30 days
- Input X: 30 days × 27 features (all numeric features)
- Target y: next-day log return (Log_Return shifted -1)
- Return HMM_Regime for each sequence (last timestep value)
- StandardScaler fit on training data only
- PyTorch Dataset and DataLoader compatible

**Key classes:**
```python
class RAMTDataset(Dataset):
    # sequences for one ticker, one fold

class RAMTDataModule:
    # handles train/val/test splits
    # fits scaler on train only
    # returns DataLoaders
```

**Success criteria:**
- Can load all 5 tickers without errors
- X shape: (batch, 30, 27)
- y shape: (batch, 1)
- regime shape: (batch,) integers 0/1/2
- Zero data leakage (scaler fit on train only)

---

### DAY 4-5 — Multimodal Encoders
**File:** models/ramt/encoder.py
**Goal:** Separate encoder per feature group

**Architecture:**
PriceEncoder (6 features → embed_dim=32):
Linear(6, 32) → LayerNorm → ReLU → Linear(32, 32)
VolatilityEncoder (5 features → embed_dim=32):
Linear(5, 32) → LayerNorm → ReLU → Linear(32, 32)
TechnicalEncoder (8 features → embed_dim=32):
Linear(8, 32) → LayerNorm → ReLU → Linear(32, 32)
MomentumEncoder (4 features → embed_dim=32):
Linear(4, 32) → LayerNorm → ReLU → Linear(32, 32)
VolumeEncoder (2 features → embed_dim=32):
Linear(2, 32) → LayerNorm → ReLU → Linear(32, 32)
RegimeEncoder (1 feature → embed_dim=32):
Embedding(3, 32)  ← HMM_Regime is integer 0/1/2
CrossAssetEncoder (1 feature → embed_dim=32):
Linear(1, 32) → LayerNorm → ReLU → Linear(32, 32)
MultimodalFusion:
Input: 7 embeddings each (batch, seq_len, 32)
Concatenate → (batch, seq_len, 224)
Linear(224, 64) → LayerNorm
Output: (batch, seq_len, 64)

**Why separate encoders:**
Each feature group has different statistical properties.
Price features are returns (small values near zero).
Volume features are ratios (larger values).
Regime is categorical (integer 0/1/2).
Separate encoders let each group normalize independently
before being fused into a unified representation.

**Success criteria:**
- Forward pass runs: input (batch, 30, 27) → output (batch, 30, 64)
- No NaN outputs
- All encoders correctly slice their feature columns

---

### DAY 6-7 — Mixture of Experts
**File:** models/ramt/moe.py
**Goal:** Regime-conditioned expert routing

**Architecture:**
ExpertTransformer (one per regime, 3 total):
TransformerEncoderLayer:
d_model=64
nhead=4
dim_feedforward=128
dropout=0.1
TransformerEncoder:
num_layers=2
Pool last timestep → Linear(64, 1)
GatingNetwork:
Input: fused_embedding (batch, 64) + regime_onehot (batch, 3)
Linear(67, 32) → ReLU → Linear(32, 3) → Softmax
Output: (batch, 3) soft weights over 3 experts
MixtureOfExperts:

Run all 3 experts independently
expert_outputs: [(batch,1), (batch,1), (batch,1)]
Stack → (batch, 3)
Get gate weights from GatingNetwork → (batch, 3)
Weighted sum: sum(gate_weights × expert_outputs)
Output: (batch, 1) final prediction


**Why soft gating instead of hard routing:**
XGBoost used hard routing — if regime=Bear, use Bear model only.
MoE uses soft gating — if regime probabilities are
Bull=10%, Bear=75%, HighVol=15%, then:
prediction = 0.10×Bull_expert + 0.75×Bear_expert + 0.15×HighVol_expert

This is more realistic because market transitions are gradual.
A stock moving from bull to bear does not flip overnight.

**Success criteria:**
- Forward pass: (batch, 30, 64) + regime (batch,) → (batch, 1)
- Gate weights sum to 1.0 for each sample
- All 3 experts produce valid outputs
- No NaN gradients

---

### DAY 8 — Full RAMT Model + Loss
**Files:** models/ramt/model.py, models/ramt/losses.py

**model.py — RAMTModel:**
Forward pass sequence:

Split input (batch, 30, 27) into 7 feature groups
MultimodalEncoder → (batch, 30, 64)
Positional encoding added
TransformerEncoder (shared backbone, 2 layers)
Extract last timestep → (batch, 64)
MoE with regime conditioning → (batch, 1)
Return prediction


**Full architecture params:**
```python
seq_len = 30
embed_dim = 64
num_heads = 4
num_transformer_layers = 2
num_experts = 3
num_regimes = 3
dropout = 0.1
```

**losses.py — Combined Loss:**
CombinedLoss:
mse_loss = MSE(y_pred, y_true)
directional_loss = mean(
relu(-(y_true × y_pred))
)
Penalizes when prediction and actual have opposite signs
relu(-product) > 0 only when signs differ
total_loss = mse_loss + lambda × directional_loss
lambda = 0.3  (tunable)

**Why combined loss:**
MSE minimizes prediction error magnitude.
But in trading, direction matters more than magnitude.
Getting the direction right on a 3% day is worth more
than perfect magnitude on a 0.1% day.
Directional loss explicitly penalizes wrong-direction predictions.

**Success criteria:**
- Full forward pass: (batch, 30, 27) → (batch, 1)
- Loss computation works
- Backward pass (gradients flow through all components)
- Model parameter count printed

---

### DAY 9-10 — Training Loop
**File:** models/train.py
**Goal:** Train RAMT on all tickers

**Training setup:**
```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # restart every 10 epochs
    T_mult=2     # double the period after each restart
)

early_stopping:
  patience = 15 epochs
  monitor = validation loss
  save best model to checkpoints/{ticker}_best.pt
```

**Training loop per ticker:**
For each ticker:
Load processed CSV
Split: 70% train, 10% val, 20% test (chronological)
Create DataLoaders (batch_size=32)
For each epoch:
Train on training batches
Compute val loss
Step scheduler
Check early stopping
Save best model
Load best model
Evaluate on test set
Print metrics

**Logging:**
- Print epoch, train_loss, val_loss every 5 epochs
- Save training curve data to results/training_curves/
- Save best model checkpoint per ticker

**Success criteria:**
- Training converges (val loss decreases then stabilizes)
- No NaN losses during training
- Best model saved for each ticker
- Training completes in reasonable time on CPU

---

### DAY 11-12 — Walk-Forward for DL
**File:** models/walk_forward_dl.py
**Goal:** Same walk-forward structure as XGBoost but for RAMT

**This is critical for fair comparison.**
XGBoost used walk-forward — RAMT must use the same.

**Walk-forward setup:**
Initial train: 60% of ticker data
Step size: 63 trading days
Test window: 63 days per fold
For each fold:

Create train/val split from current training window
(last 15% of training = validation)
Fit StandardScaler on training data only
Initialize fresh RAMT model
Train with early stopping on validation loss
Predict on test fold
Collect predictions
Advance window by 63 days

After all folds:
Pool all predictions
Compute RMSE, MAE, DA%, Sharpe
Also compute: Max Drawdown, Profit Factor, Calmar Ratio

**New metrics to add:**
```python
def max_drawdown(strategy_returns):
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def profit_factor(strategy_returns):
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    return gains / losses if losses > 0 else float('inf')

def calmar_ratio(strategy_returns):
    annual_return = strategy_returns.mean() * 252
    mdd = abs(max_drawdown(strategy_returns))
    return annual_return / mdd if mdd > 0 else float('inf')
```

**Output:**
- results/ramt_predictions.csv
- Same format as xgboost_predictions.csv and lstm_predictions.csv

**Success criteria:**
- Walk-forward completes for all 5 tickers
- Zero data leakage (scaler refit every fold)
- All 7 metrics computed
- Predictions CSV saved

---

### DAY 13 — Full Comparison Table
**File:** models/evaluate_all.py
**Goal:** Compare all models side by side

**Output table format:**
ModelTickerRMSEMAEDA%SharpeMaxDDCalmarXGBoostJPM0.01940.012752.130.52-0.121.45XGBoostRELIANCE_NS0.01800.012152.250.54-0.091.67...LSTMJPMx.xxxxx.xxxxxx.xxx.xx-x.xxx.xx...RAMTJPMx.xxxxx.xxxxxx.xxx.xx-x.xxx.xx...--------------------------------------------------------------------XGBoostAVERAGE0.01870.012652.130.27......LSTMAVERAGEx.xxxxx.xxxxxx.xxx.xx......RAMTAVERAGEx.xxxxx.xxxxxx.xxx.xx......

**Save to:** results/full_comparison.csv

**Also generate:**
- Per-ticker improvement: RAMT vs XGBoost delta
- Best model per ticker
- Statistical significance note

---

### DAY 14 — GitHub + Report Update
**Tasks:**
1. Update README with Phase 2 results table
2. Update PHASE2_PLAN.md with actual achieved results
3. Commit all new files:
git add models/ramt/
git add models/baseline_lstm.py
git add models/train.py
git add models/walk_forward_dl.py
git add models/evaluate_all.py
git add results/
git commit -m "phase2: LSTM + RAMT implementation with walk-forward evaluation"
git push origin main
4. Share final results table with Vivek for report update

### DAY 15 — Buffer
- Fix any broken components
- Rerun any ticker that gave unexpected results
- Polish code documentation
- Prepare viva answers for DL components

---

## Key Design Decisions (Justify These in Viva)

### Why seq_len=30?
30 trading days = 6 weeks.
Captures one monthly options cycle.
Long enough for momentum patterns.
Short enough to avoid stale information.
Validated by ACF/PACF showing autocorrelation dies at ~20-30 lags.

### Why embed_dim=64?
27 input features → 64 embedding.
Enough capacity to capture cross-feature interactions.
Small enough to train on CPU in reasonable time.
Standard practice for small financial datasets.

### Why 3 experts in MoE?
Matches our 3 HMM regimes exactly.
Bull expert, Bear expert, HighVol expert.
One-to-one alignment between regime detection and expert routing.

### Why AdamW over Adam?
AdamW decouples weight decay from gradient updates.
Better regularization for transformer models.
Standard choice in all modern transformer training.

### Why Cosine Annealing with Warm Restarts?
Financial data has non-stationary patterns.
Warm restarts help escape local minima corresponding to specific regimes.
Allows learning rate to explore multiple loss landscape regions.

### Why Combined Loss (MSE + Directional)?
MSE alone optimizes for magnitude accuracy.
But trading profit depends on direction, not magnitude.
A prediction of +0.1% when actual is +3% is directionally correct —
MSE penalizes this but trading profits from it.
Lambda=0.3 gives 70% weight to magnitude and 30% to direction.

---

## Baseline Results to Beat (From Phase 1)
XGBoost Regime-Stratified (current best):
JPM:         RMSE=0.0194, DA=52.13%, Sharpe=0.52
RELIANCE_NS: RMSE=0.0180, DA=52.25%, Sharpe=0.54
TCS_NS:      RMSE=0.0151, DA=53.44%, Sharpe=0.82
HDFCBANK_NS: RMSE=0.0165, DA=51.52%, Sharpe=0.04
EPIGRAL_NS:  RMSE=0.0243, DA=51.32%, Sharpe=-0.56
Average:     RMSE=0.0187, DA=52.13%, Sharpe=0.27
RAMT must beat:
DA% > 54% average
Sharpe > 1.0 average
RMSE < 0.0173 average

---

## Dependencies Required
Add to requirements.txt if not present
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
hmmlearn>=0.3.0
matplotlib>=3.7.0
seaborn>=0.13.0

## Environment Check Before Starting

Run this every time before coding:
```bash
cd regime-adaptive-transformer
source .venv/bin/activate
python -c "
import torch
import pandas
import numpy
import sklearn
print('torch:', torch.__version__)
print('pandas:', pandas.__version__)
print('numpy:', numpy.__version__)
print('sklearn:', sklearn.__version__)
print('CUDA:', torch.cuda.is_available())
print('All good — ready to code')
"
```

---

## Notes for Cursor

When implementing any file:
1. Always import from relative paths within the project
2. Always use device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
3. Always set random seeds: torch.manual_seed(42), numpy random seed 42
4. Always print model parameter count after instantiation
5. Feature column order must match exactly:
   Price cols first, then Vol, Tech, Momentum, Volume, Regime, CrossAsset
6. Target variable always: y = Log_Return.shift(-1), drop last row
7. Walk-forward scaler always fit on training window only — never on test data
8. Save all predictions with columns: Date, Ticker, y_true, y_pred

---

*Last updated: Phase 2 start*
*XGBoost baseline: DA=52.13%, Sharpe=0.27*
*RAMT target: DA>54%, Sharpe>1.0*