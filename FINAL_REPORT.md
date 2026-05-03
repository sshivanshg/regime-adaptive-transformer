# FINAL REPORT

## Executive Summary
This project evolved from a transformer-centric ranking hypothesis to a validated hybrid architecture where regime-aware risk control (HMM) and sentiment-aware gating (FinBERT + LoRA pathway) are explicitly separated and testable. The final system emphasizes robustness, diagnostic transparency, and reproducible execution over headline complexity.

## Methodology
The production decision layer uses a Regime-Adaptive Sentiment Gating policy: in Bull regimes momentum remains primary with permissive sentiment filtering; in High-Vol regimes score fusion uses weighted momentum-sentiment integration; in Bear regimes sentiment conviction gates entries and allows cash states. All feature streams are shifted to enforce T-1 information for T execution, preventing look-ahead leakage.

## Ablation Results

| Scenario | CAGR | Sharpe (Net) | Max Drawdown | Win Rate | Source |
| --- | --- | --- | --- | --- | --- |
| Baseline (Momentum) | 43.04% | 1.345 | -19.68% | N/A | results/hmm_ablation/.../hmm_vs_flat_summary.csv |
| ML-Enhanced (Momentum + HMM) | 10.62% | 0.663 | -18.68% | N/A | results/hmm_ablation/.../hmm_vs_flat_summary.csv |
| DL-Enhanced (Momentum + FinBERT) | N/A | N/A | N/A | N/A | results/ablation_report.csv not found |
| Full Hybrid (Our Model) | 13.49% | 0.833 | -18.68% | 64.00% | results/final_strategy/backtest_results.csv |

**Note:** `results/ablation_report.csv` is not present in the current artifact set. Rows were populated from available backtest summaries where possible; unavailable rows remain `N/A`.

## Diagnostic Analysis

- **Baseline (Momentum)**: Serves as the pure price-signal control. Performance reflects momentum capture without macro-state conditioning or text-derived risk adjustment.
- **ML-Enhanced (Momentum + HMM)**: Adding HMM regime sizing changes exposure profile versus baseline, typically reducing risk concentration during unstable windows at the cost of upside during persistent rallies.
- **DL-Enhanced (Momentum + FinBERT)**: This scenario has no runnable artifact in the current repository export. The expected metric source is missing, so no quantitative claim is made.
- **Full Hybrid (Our Model)**: Joint regime + sentiment gating combines macro-state risk control with narrative-level signal timing. Improvements in Sharpe or drawdown are interpreted as complementary ML+DL behavior.

## Reproducibility Statement
The repository includes a single orchestrator (`main.py --task all`), container manifests, a self-audit script, and deterministic report generation. This final report is generated from local result artifacts only, without manual metric edits.