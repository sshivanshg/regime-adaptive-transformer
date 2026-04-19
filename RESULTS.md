# Results log

Single place to record **measured outcomes** from experiments (backtests, ablations, benchmarks). Update this file when you add a new run; point to CSVs or figures under `results/` rather than duplicating large tables.

**Convention:** Each subsection states *what differed*, *the test window*, *metrics*, and *where to reproduce*.

---

## HMM regime-sizing ablation (momentum vs RAMT): four OOS windows

**What this compares:** Same ranking inputs per row, two portfolio rules — **HMM-conditioned** (`run_backtest_daily` default: NIFTY `HMM_Regime` sleeves + regime top‑N) vs **flat regime sizing** (`flat_regime_sizing=True`: full 1.0/1.0/1.0, `top_n` in every regime). Not a neural-net ablation.

### Consolidated table (chronological)

| Window | Signal / universe | Equities OK (requested) | HMM Sharpe | HMM CAGR | HMM Max DD | Flat Sharpe | Flat CAGR | Flat Max DD | One-line context |
|--------|-------------------|-------------------------|------------|----------|------------|-------------|-----------|-------------|------------------|
| 2008–01-01 — 2010–12-31 | Momentum `Ret_21d`; **NIFTY 200** (current NSE CSV at fetch) | **~138** (~200) | 0.25 | 3.7% | -43.2% | 0.17 | 0.8% | -52.7% | GFC / early recovery |
| 2010–01-01 — 2012–12-31 | Momentum `Ret_21d`; **NIFTY 100 proxy** (`scripts/universe/nifty100_nse_survivorship_proxy.txt`) | **78** (100) | 0.15 | 0.8% | -12.2% | -0.11 | -3.0% | -30.1% | Post-GFC recovery; later EM stress / taper talk |
| 2013–01-01 — 2015–12-31 | Momentum `Ret_21d`; same **NIFTY 100 proxy** | **79** (100) | 0.16 | 0.8% | -33.2% | 0.15 | 0.6% | -33.2% | Pre-demonetization; India bull with volatility |
| 2024–01-01 — 2025–12-31 | **RAMT** `ranking_predictions.csv` (not momentum) | n/a (full OOS pipeline) | 0.66 | 10.6% | -18.7% | 1.35 | 43.0% | -19.7% | Modern RAMT blind-test window |

**Artifacts (per historical window):** `data/raw_yf_<tag>/` (includes `_fetch_stats.json` after fetch), `data/processed_yf_<tag>/`, `results/momentum_rankings_yf_<tag>.csv`, `results/hmm_vs_flat/yf_<tag>/<start>_<end>/hmm_vs_flat_summary.csv`.

**Reproduce historical chain (example 2013–2015):**

```bash
. .venv/bin/activate
python scripts/run_yf_hmm_ablation.py \
  --start 2013-01-01 --end-exclusive 2016-01-01 \
  --bt-start 2013-01-01 --bt-end 2015-12-31 --tag 2013_2015 \
  --universe-file scripts/universe/nifty100_nse_survivorship_proxy.txt
```

`fetch_nifty200.py` flags: `--universe-file` (one `SYMBOL.NS` per line), or `--index nifty50|nifty100|nifty200|nifty500`. After each download, read **`data/raw_yf_<tag>/_fetch_stats.json`** for equity OK vs failed tickers.

### Caveats (read before citing)

1. **Survivorship / universe:** Historical rows use **current** NSE index CSVs or the checked-in **NIFTY 100 proxy** — **not** point-in-time 2008/2010/2013 membership. All three historical windows are **survivorship-biased** relative to a true period index. The **2024–2025** column uses **RAMT** predictions, not momentum — compare cross-window levels qualitatively, not as a single homogeneous experiment.
2. **Macro features in `feature_engineering`:** Same pipeline for all: INDIAVIX, crude, USDINR, SP500 lagged 1d. **INDIAVIX** has limited/fragile history in the 2008 era on Yahoo; rows are forward-filled / zeroed as in code. Not identical macro coverage quality across decades.
3. **Failed downloads:** Many **NIFTY 100** names IPO’d after 2010 or have no Yahoo history for the window — see `_fetch_stats.json` failed lists (typically **~21–22** failed per 2010–2015 run; **2008–2010** used a wider NSE list with **~62** failed from an earlier full fetch).
4. **Stops:** Per-stock stops in `run_backtest_daily` do not yet cut realized returns (audit flags only); see `results/sensitivity/parameter_sensitivity_meta.json`.

### Earlier single-window notes (parameter sensitivity, 2024–2026 sensitivity grid)

| Configuration | Sharpe | CAGR | Max DD | Notes |
|----------------|--------|------|--------|--------|
| HMM sizing on (baseline rules) | 0.83 | 13.5% | -18.7% | Sector cap on, top-5, SL 7% (bear 5%), Kelly + friction as in `run_final_2024_2026.py` |
| Flat 1.0/1.0/1.0 (no regime sizing) | 1.39 | 42.5% | -19.7% | Same predictions; full risk-on every regime |

**Test window:** 2024-01-01 — 2026-04-16 (blind test slice, `Period == "Test"`).

**Artifacts:** `results/sensitivity/parameter_sensitivity_summary.csv`. **Reproduce:** `python scripts/parameter_sensitivity_backtest.py`

### What to call the HMM vs flat comparison

| Name | Meaning |
|------|--------|
| **HMM-conditioned portfolio** vs **regime-agnostic portfolio** | HMM drives risk sleeves vs flat sizing. |
| **Regime sizing ablation** | “HMM on” vs “HMM off” in *portfolio rules* (not in the neural net). |

---

## Parameter sensitivity (hyperparameter grid)

One-off grid: `top_n`, stop-loss, sector cap, flat regime sizing. Summary table and per-variant backtest CSVs live under **`results/sensitivity/`**.

**Caveat:** In the current `run_backtest_daily` implementation, per-stock stops flag `stops_hit` but **realized window returns use the full price path**, so varying `stop_loss` alone may not change PnL until stop-outs are applied to returns. See `results/sensitivity/parameter_sensitivity_meta.json`.

---

## Other results (add rows as you go)

| Study | Metric(s) | Location | Date |
|-------|-----------|----------|------|
| Main backtest export | Per-window returns | `results/backtest_results.csv` | (update when regenerated) |
| Walk-forward training history | Loss / fold | `results/training_history.csv`, `results/training_dashboard.png` | |
| Ranking / predictions | Cross-sectional scores | `results/ranking_predictions.csv` | |

---

## Related docs

- `FEATURES_AND_REGIMES.md` — what goes into `X`, how `HMM_Regime` is built, how it is used in the portfolio layer.
- `README.md` — project narrative and pitfalls.
