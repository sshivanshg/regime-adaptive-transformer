# Phase 1 Literature Review (Rubric-Aligned, Simple Explanation)

## Introduction
This project studies short-horizon equity forecasting for one US stock (JPM) and three Indian stocks (RELIANCE, TCS, HDFCBANK). The practical challenge is not just prediction error; it is adaptation. Equity behavior changes across calm periods, crisis periods, and policy transitions, so a model that assumes one fixed market law usually degrades when regimes shift.

Our repository already reflects this problem framing. `eda/eda.ipynb` shows heavy tails, volatility clustering, and time-varying rolling correlations. `features/feature_engineering.py` adds regime labels and cross-asset context. `models/baseline_lstm.py` and `models/baseline_xgboost.py` provide strict chronological baselines. `models/ramt/*` implements a regime-aware Transformer with mixture-of-experts (MoE).

**Gap Argument (Rubric Level-5):** prior methods each solve only part of the problem. Classical models provide structure but fail under regime breaks. LSTMs model sequence nonlinearity but still encode one dominant process. Transformer-only models improve temporal coverage but do not automatically separate market states. SOTA multimodal systems align news with prices, and SOTA hybrid T-GNN systems model dependence structure, yet a unified cross-market framework that uses **explicit HMM-derived regime labels** to gate a **multimodal Transformer+MoE** for US-India transfer remains underdeveloped. RAMT is designed to fill this exact gap.

---

## 1) Failure of Classical Statistics (Basic)

### Why ARIMA/GARCH break in practice
ARIMA and GARCH are useful references, but they assume that the underlying process is stable enough to be summarized by one compact parametric form over long windows. In equity markets, this assumption is fragile during event shocks such as COVID-19 and post-2022 monetary tightening cycles.

### Evidence from this project
From `eda/eda.ipynb`:
- Return kurtosis is far above Gaussian 3 (examples: JPM ~16.03, RELIANCE ~11.44, HDFCBANK ~13.83, TCS ~6.70).
- Volatility is clustered, not uniform through time.
- 60-day rolling stock-index correlation is non-static and regime-sensitive.

So even when ADF indicates stationarity in log returns (unit-root sense), higher moments and dependency patterns are still unstable in the modeling sense. This is the operational meaning of **Non-Stationarity** for forecasting models.

---

## 2) Rigidity of RNNs/LSTMs (Intermediate)

### Why LSTMs are necessary but limited
Your `models/baseline_lstm.py` is well-implemented as a Phase-1 benchmark:
- chronological train/val/test,
- next-day log-return regression,
- early stopping and gradient clipping,
- RMSE/MAE plus directional accuracy and Sharpe.

But LSTM carries a strong **Inductive Bias**: one recurrent parameterization is expected to explain all states. For cross-market equities, the signal-to-noise ratio is already low, and regime-dependent behavior is strong. A single recurrent dynamics model often averages incompatible behaviors (calm vs panic, domestic vs external shock transmission).

### Practical limitation
LSTM can learn nonlinear memory, but it does not explicitly route computation by market state unless we add a separate control mechanism.

---

## 3) Power of Attention (Advanced)

### Why Transformers matter
Attention-based models address key limitations of recurrence:
- direct long-range dependency access,
- parallel training,
- better handling of sparse event-driven effects,
- easier extension to multimodal and expert-routing designs.

This follows the core argument of Vaswani et al. (2017). PatchTST (2023) further improves time-series forecasting by chunking sequences into local patches, helping models represent multi-scale structure more cleanly.

### Connection to RAMT code
In `models/ramt/model.py`, a Transformer encoder forms the temporal backbone. In `models/ramt/encoder.py`, features are grouped and projected by modality before fusion, so the temporal model receives cleaner, semantically partitioned inputs.

---

## 4) Intelligence of Multimodality (SOTA)

### Why price-only modeling is incomplete
Price carries delayed and filtered information. News and sentiment can move expectations before full price adjustment, especially during stress periods.

### FinBERT and LLM direction
FinBERT-style modeling shows that domain-adapted language encoders can extract financial sentiment signals more reliably than generic text pipelines. The ICLR 2026 paper by Ying et al. pushes this further by combining LLM features with Transformer-based temporal modules and volatility-aware decision components. Their framework also highlights **Attention Reprogramming** as a bridge between numeric price representations and language-semantic space.

### Relevance to this project
Even though current Phase-1 RAMT training is mainly numerical-feature-driven, the architectural direction is aligned with multimodal extension: modular encoding, temporal attention backbone, and adaptive routing mechanisms.

---

## 5) Connectivity of GNNs (SOTA)

### Why relational structure matters
Stocks are not independent channels. Correlation, co-movement, and market-linkage transmission are structural parts of the forecasting problem.

Fanshawe et al. (2025) show this clearly through a hybrid Transformer-GNN setup for forward correlation forecasting in **Fisher-z space**. Their results support the claim that relation-aware modeling improves downstream decisions compared with static dependence assumptions.

### Why this supports US-India context
Your project explicitly uses cross-market context through rolling stock-index linkage (`Rolling_Corr_Index`) with `^GSPC` (S\&P 500) for JPM and `^NSEI` (NIFTY50) for Indian names in `features/feature_engineering.py`. This design choice is consistent with the connectivity argument: market-level dependence is dynamic and must be represented as a time-varying signal, not a fixed scalar.

---

## 6) How the Repository Implements the Literature Logic

## 6.1 Feature design and regime logic
`features/feature_engineering.py` creates seven feature groups:
- lagged returns,
- volatility,
- technical indicators,
- momentum,
- volume,
- HMM regime labels,
- rolling cross-asset market correlation.

The HMM block uses three latent states over `[Log_Return, Realized_Vol_20]`, then maps them semantically into `bull`, `bear`, and `high_vol` regimes. This is explicit regime supervision, not only latent feature drift.

## 6.2 Baselines for scientific control
- `models/baseline_lstm.py`: sequence baseline.
- `models/baseline_xgboost.py`: walk-forward tabular baseline with regime-stratified routing plus global fallback.

This baseline structure is important for rubric quality because model choice is justified by comparison, not by trend.

## 6.3 RAMT specialization mechanism
In `models/ramt/moe.py`, gating logits are shifted by regime-specific bias (`regime_bias`), so expert weights depend on the inferred market state. This is the key adaptation mechanism:
- same temporal embedding,
- different expert mixture under bull/bear/high-vol.

In `models/ramt/model.py`, the sequence is:
1. extract regime from the latest timestep,
2. multimodal encode,
3. temporal Transformer,
4. regime-conditioned MoE,
5. return prediction head.

---

## 7) Rubric Check (Basic to Advanced)

| Rubric Requirement | Covered Here | Repository Anchor |
|---|---|---|
| Failure of Classical Stats | Section 1 | `eda/eda.ipynb` |
| Rigidity of RNNs/LSTMs | Section 2 | `models/baseline_lstm.py` |
| Power of Attention (Transformer + PatchTST) | Section 3 | `models/ramt/model.py`, `models/ramt/encoder.py` |
| Intelligence of Multimodality (FinBERT + ICLR 2026) | Section 4 | SOTA alignment + RAMT modular design |
| Connectivity of GNNs (T-GNN 2025) | Section 5 | `Rolling_Corr_Index` in `features/feature_engineering.py` |
| Explicit Gap Argument linked to current needs | **Last paragraph of Introduction** | Directly stated |

---

## 8) Final Position
RAMT is justified not because it is more complex, but because each added component answers a concrete failure mode observed in data and in prior methods:
- heavy tails and regime shifts require state awareness,
- cross-market noise requires robust temporal modeling,
- relation dynamics require dependence-aware context,
- regime-conditioned routing is needed to avoid one-size-fits-all dynamics.

This is the strongest Phase-1 narrative: implement baselines, show empirical instability patterns, identify gaps in literature, and justify RAMT as a targeted architectural response.

---

## References (for report bibliography)
1. Vaswani et al. (2017), *Attention Is All You Need*.
2. Nie et al. (2023), *PatchTST: A Time Series is Worth 64 Words*.
3. Fanshawe et al. (2025), *Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network*.
4. Ying et al. (2026), *Aligning News and Prices: A Cross-Modal LLM-Enhanced Transformer DRL Framework for Volatility-Adaptive Stock Trading*.
5. Hochreiter and Schmidhuber (1997), *Long Short-Term Memory*.
6. Box-Jenkins ARIMA foundations; Engle (1982) ARCH; Bollerslev (1986) GARCH.
7. FinBERT literature for financial-sentiment representation.
