from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from models.backtest import run_backtest_daily

logger = logging.getLogger("hybrid_backtester")

def _normalize_ticker_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(".", "_", regex=False)

def build_triple_expert_predictions(
    momentum_df: pd.DataFrame,
    chronos_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    mode: str = "triple_expert"
) -> pd.DataFrame:
    """
    Implements Regime-Adaptive Fusion Logic for Triple-Expert System.
    1. Technical Expert: Ret_21d (Momentum)
    2. Foundation Expert: Chronos-T5 (LoRA)
    3. Risk Expert: HMM Regime
    """
    # 1. Align and Merge
    m = momentum_df[['Date', 'Ticker', 'Ret_21d', 'actual_alpha']].copy()
    c = chronos_df[['Date', 'Ticker', 'predicted']].rename(columns={'predicted': 'chronos_score'}).copy()
    r = regime_df[['Date', 'regime']].copy()

    # Ensure temporal alignment (T-1 signals for T execution)
    m['Date'] = pd.to_datetime(m['Date']).dt.normalize()
    c['Date'] = pd.to_datetime(c['Date']).dt.normalize()
    r['Date'] = pd.to_datetime(r['Date']).dt.normalize()

    m['Ticker'] = _normalize_ticker_series(m['Ticker'])
    c['Ticker'] = _normalize_ticker_series(c['Ticker'])

    # Shift Chronos and Momentum to T-1 to ensure no leakage
    m = m.sort_values(['Ticker', 'Date'])
    c = c.sort_values(['Ticker', 'Date'])
    
    m['momentum_t1'] = m.groupby('Ticker')['Ret_21d'].shift(1)
    m['actual_alpha_t'] = m['actual_alpha'] # Current T actual alpha for metrics
    
    c['chronos_t1'] = c.groupby('Ticker')['chronos_score'].shift(1)
    
    # Shift Regime to T-1
    r = r.sort_values('Date')
    r['regime_t1'] = r['regime'].shift(1)

    # Merge
    df = m.merge(c[['Date', 'Ticker', 'chronos_t1']], on=['Date', 'Ticker'], how='inner')
    df = df.merge(r[['Date', 'regime_t1']], on='Date', how='inner')
    
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Ticker', 'predicted_alpha', 'actual_alpha'])

    # Triple-Expert Adaptive Fusion
    def calculate_fused_score(row):
        regime = int(row['regime_t1'])
        mom = float(row['momentum_t1'])
        chr = float(row['chronos_t1'])
        
        if mode == "baseline": # Momentum + HMM (Production Baseline)
            res = mom
        elif mode == "foundation_only": # Chronos only
            res = chr
        elif mode == "simple_hybrid": # 50/50
            res = 0.5 * mom + 0.5 * chr
        elif regime == 1: # Bull: 70% Momentum / 30% Chronos
            res = 0.7 * mom + 0.3 * chr
        elif regime == 0: # Volatile: 30% Momentum / 70% Chronos
            res = 0.3 * mom + 0.7 * chr
        elif regime == 2: # Bear: 10% Momentum / 90% Chronos
            res = 0.1 * mom + 0.9 * chr
        else:
            res = 0.5 * mom + 0.5 * chr
        return res

    # Ensure we get a Series of floats
    fused_scores = df.apply(calculate_fused_score, axis=1)
    df['predicted_alpha'] = fused_scores.values.astype(float)
    df['actual_alpha'] = df['actual_alpha_t']
    
    # Final check for nans
    df = df.dropna(subset=['predicted_alpha', 'actual_alpha'])
    
    # Return formatted for backtest_daily
    return df[['Date', 'Ticker', 'predicted_alpha', 'actual_alpha']].sort_values(['Date', 'predicted_alpha'], ascending=[True, False])

def run_diagnostic_backtest(
    name: str,
    mode: str,
    processed_dir: Path,
    chronos_preds_path: Path,
    nifty_features_path: Path,
    raw_dir: Path,
    start: str,
    end: str,
    capital: float = 100000.0
):
    """
    Wrapper for running backtests in the ablation study.
    """
    # Load Data
    # Note: Using build_ticker_universe to get all available stocks
    from models.ramt.dataset import build_ticker_universe
    tickers = build_ticker_universe(str(processed_dir))
    
    mom_chunks = []
    for t in tickers:
        p = processed_dir / f"{t}_features.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=['Date', 'Ticker', 'Ret_21d', 'Sector_Alpha', 'Monthly_Alpha'])
            target = 'Sector_Alpha' if df['Sector_Alpha'].notna().any() else 'Monthly_Alpha'
            df = df.rename(columns={target: 'actual_alpha'})
            mom_chunks.append(df[['Date', 'Ticker', 'Ret_21d', 'actual_alpha']])
    momentum_panel = pd.concat(mom_chunks)
    
    chronos_panel = pd.read_csv(chronos_preds_path)
    regime_panel = pd.read_parquet(nifty_features_path, columns=['Date', 'HMM_Regime']).rename(columns={'HMM_Regime': 'regime'})

    # Build Predictions
    preds = build_triple_expert_predictions(momentum_panel, chronos_panel, regime_panel, mode=mode)
    
    # Filter by dates
    preds = preds[(preds['Date'] >= pd.to_datetime(start)) & (preds['Date'] <= pd.to_datetime(end))]

    # Run Daily Backtest with Standard Guardrails
    bt_results = run_backtest_daily(
        predictions_df=preds,
        nifty_features_path=str(nifty_features_path),
        raw_dir=str(raw_dir),
        start=start,
        end=end,
        top_n=5,
        capital=capital,
        stop_loss=0.07,
        rebalance_friction_rate=0.0022,
        use_sector_cap=True,
        flat_regime_sizing=(mode in ["foundation_only", "simple_hybrid"]) # Disable HMM sizing for these
    )
    
    return bt_results, preds
