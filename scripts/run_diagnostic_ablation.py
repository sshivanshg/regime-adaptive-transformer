from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hybrid_backtester import run_diagnostic_backtest
from models.ablation_engine import compute_metrics

def run_diagnostic_study():
    print("🚀 Starting 4-Way Diagnostic Ablation Study...")
    
    processed_dir = ROOT / "data" / "processed"
    chronos_preds = ROOT / "results" / "lora" / "lora_predictions.csv"
    nifty_features = ROOT / "data" / "processed" / "_NSEI_features.parquet"
    raw_dir = ROOT / "data" / "raw"
    
    start_date = "2024-01-01"
    end_date = "2026-04-15"
    
    scenarios = [
        ("Baseline (ML): Momentum + HMM", "baseline"),
        ("Foundation Only (DL): Chronos-T5 (LoRA)", "foundation_only"),
        ("Simple Hybrid: Momentum + Chronos (50/50)", "simple_hybrid"),
        ("Triple-Expert (Proposed): Momentum + Chronos + HMM", "triple_expert")
    ]
    
    results = []
    all_bt_data = {}

    for name, mode in scenarios:
        print(f"\nEvaluating: {name}...")
        bt_df, preds_df = run_diagnostic_backtest(
            name=name,
            mode=mode,
            processed_dir=processed_dir,
            chronos_preds_path=chronos_preds,
            nifty_features_path=nifty_features,
            raw_dir=raw_dir,
            start=start_date,
            end=end_date
        )
        
        print(f"DEBUG: bt_df rows={len(bt_df)}")
        if not bt_df.empty:
            print(f"DEBUG: bt_df columns={bt_df.columns.tolist()}")
            print(f"DEBUG: last nav={bt_df['portfolio_value'].iloc[-1]}")
        
        metrics = compute_metrics(bt_df, capital=100000.0)
        metrics["Scenario"] = name
        results.append(metrics)
        all_bt_data[name] = bt_df

    # --- Generate Report ---
    report_df = pd.DataFrame(results)
    
    # Calculate "LoRA Alpha" (Improvement over Baseline)
    baseline_cagr = report_df[report_df['Scenario'] == "Baseline (ML): Momentum + HMM"]['CAGR'].values[0]
    proposed_cagr = report_df[report_df['Scenario'] == "Triple-Expert (Proposed): Momentum + Chronos + HMM"]['CAGR'].values[0]
    lora_alpha = proposed_cagr - baseline_cagr
    
    print("\n" + "="*80)
    print("DIAGNOSTIC ABLATION SUMMARY")
    print("="*80)
    print(report_df[['Scenario', 'CAGR', 'Sharpe_Net', 'Max_Drawdown']].to_string(index=False))
    print("\n" + "="*80)
    print(f"Triple-Expert Synergy (LoRA Alpha): {lora_alpha*100:+.2f}% vs Production Baseline")
    print("="*80)

    # Save Results
    output_path = ROOT / "results" / "ablation_summary.json"
    report_df.to_json(output_path, orient='records', indent=4)
    print(f"\nDetailed report saved to {output_path}")

if __name__ == "__main__":
    run_diagnostic_study()
