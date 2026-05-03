# 5-Minute Presentation Script

## 0:00 - 0:40 | Problem and Motivation
- We built a Regime-Adaptive Transformer project for NIFTY 200 stock ranking under non-stationary market conditions.
- Core challenge: momentum alone is strong but fragile during high-volatility and drawdown regimes.
- Objective: prove whether a hybrid HMM + sentiment system can improve robustness while staying reproducible.

## 0:40 - 1:35 | Synergistic Innovation
- Show [docs/architecture_final.png](architecture_final.png).
- Explain the three-lane architecture:
  - ML lane: momentum signal + HMM regime state.
  - DL lane: FinBERT sentiment with LoRA adaptation.
  - Fusion lane: regime-adaptive gating logic.
- Key statement: this is not feature stuffing; it is conditional decision logic where sentiment has different roles by regime.

## 1:35 - 2:35 | Diagnostic Ablation
- Open [results/ablation_report.csv](../results/ablation_report.csv) and summarize four scenarios:
  - Baseline (Momentum)
  - ML-Enhanced (Momentum + HMM)
  - DL-Enhanced (Momentum + FinBERT)
  - Full Hybrid (Momentum + HMM + FinBERT + LoRA)
- Emphasize causality: each toggle isolates one system component.
- Mention sentiment alpha by regime from [results/ablation/ablation_sentiment_alpha_by_regime.csv](../results/ablation/ablation_sentiment_alpha_by_regime.csv) when available.

## 2:35 - 3:40 | Extra Mile Deliverables
- Show dashboard Phase 3 interactive section in [dashboard/app.py](../dashboard/app.py):
  - Sentiment Explorer (ticker-level price + sentiment)
  - Regime-sentiment heatmap
  - Live ablation toggle curves
  - Explainability snapshot for top picks
- Show explainability report from [results/case_study_explainability.pdf](../results/case_study_explainability.pdf).
- Explain Integrated Gradients token attribution and regime sensitivity analysis.

## 3:40 - 4:30 | Reproducibility and Audit
- Run one command: `python main.py --task all --config config/hybrid_config.yaml`.
- Show turn-key assets: [Dockerfile](../Dockerfile), [docker-compose.yml](../docker-compose.yml), [run_all.sh](../run_all.sh).
- Show self-grading output from [SUBMISSION_CHECKLIST.txt](../SUBMISSION_CHECKLIST.txt).

## 4:30 - 5:00 | Conclusion
- Technical outcome: a modular, testable architecture with explicit ablations and reproducibility checks.
- Research outcome: robust methodology with transparent failure modes and measurable improvements.
- Final note: the repository is packaged for grader execution and publication-style review.
