from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def check_alignment():
    print("Checking Pipeline Health & Data Alignment...")
    
    # 1. Check Chronos Predictions
    preds_path = ROOT / "results" / "lora" / "lora_predictions.csv" # The default output from train_lora.py
    if not preds_path.exists():
        # Check if v2 path exists
        preds_path = ROOT / "results" / "lora" / "lora_v2_predictions.csv"
        
    if not preds_path.exists():
        print("❌ FAILED: Chronos predictions not found.")
        return False
        
    df = pd.read_csv(preds_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Chronos Predictions found: {len(df)} rows.")
    print(f"   Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # 2. Check for Overlap with Backtest Dates (2024-2026)
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2026-04-15")
    
    mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
    coverage = df[mask]
    
    if len(coverage) == 0:
        print(f"❌ FAILED: No predictions found in test window {test_start.date()} to {test_end.date()}.")
        return False
    
    print(f"✅ Data Coverage: Found {len(coverage)} rows in test window.")
    
    # 3. Check for duplicates
    dupes = df.duplicated(subset=['Date', 'Ticker']).sum()
    if dupes > 0:
        print(f"⚠️ WARNING: Found {dupes} duplicate Date/Ticker pairs in predictions.")
    else:
        print("✅ No duplicates found in prediction file.")

    print("\n🚀 Pipeline health is OK. Ready for Diagnostic Ablation.")
    return True

if __name__ == "__main__":
    if not check_alignment():
        sys.exit(1)
