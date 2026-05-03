from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.lora_experiment.chronos_lora_v2 import ChronosLoRARankerV2
from models.ramt.dataset import ALL_FEATURE_COLS, build_ticker_universe

"""
User Story: As a researcher, I want to understand which features the Chronos-T5 model prioritizes 
to ensure its "Foundation Expert" signals are grounded in meaningful market data.

Approach: Implement Permutation Feature Importance to quantify the contribution of each of the 
10 RAMT features to the model's alpha predictions on the test set.
"""

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 30

def load_model():
    model_path = ROOT / "models" / "lora_experiment" / "chronos_v2_adapter.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = ChronosLoRARankerV2(input_dim=len(ALL_FEATURE_COLS))
    checkpoint = torch.load(model_path, map_location="cpu")
    model.encoder.load_state_dict(checkpoint['encoder_lora_state_dict'])
    model.input_projection.load_state_dict(checkpoint['input_projection_state_dict'])
    model.ranking_head.load_state_dict(checkpoint['ranking_head_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

def get_test_batch(processed_dir: Path, n_tickers: int = 20):
    tickers = build_ticker_universe(str(processed_dir))[:n_tickers]
    all_x = []
    
    for ticker in tickers:
        p = processed_dir / f"{ticker}_features.parquet"
        if not p.exists(): continue
        df = pd.read_parquet(p)
        df = df.sort_values("Date").tail(100) # Use recent data for explanation
        if len(df) < SEQ_LEN: continue
        
        feats = df[list(ALL_FEATURE_COLS)].to_numpy(dtype=np.float32)
        for i in range(SEQ_LEN, len(feats)):
            all_x.append(feats[i-SEQ_LEN:i])
            
    return torch.tensor(np.array(all_x)).to(DEVICE)

def permutation_importance(model, x_orig, feature_names):
    print("Calculating Permutation Importance...")
    with torch.no_grad():
        baseline_preds = model(x_orig).cpu().numpy()
        
    importances = {}
    
    for i, col in enumerate(tqdm(feature_names, desc="Features")):
        x_perm = x_orig.clone()
        # Permute the feature across all samples and all time steps in the sequence
        # x_perm shape: (batch, seq_len, n_features)
        perm_idx = torch.randperm(x_perm.size(0))
        x_perm[:, :, i] = x_perm[perm_idx, :, i]
        
        with torch.no_grad():
            perm_preds = model(x_perm).cpu().numpy()
            
        # Importance = mean absolute difference in predictions
        # (How much did the predictions change when we destroyed this feature?)
        importance = np.mean(np.abs(baseline_preds - perm_preds))
        importances[col] = float(importance)
        
    return importances

def main():
    print("🚀 Starting Chronos-T5 Explainability Module...")
    
    model = load_model()
    processed_dir = ROOT / "data" / "processed"
    x_test = get_test_batch(processed_dir)
    
    importances = permutation_importance(model, x_test, list(ALL_FEATURE_COLS))
    
    # Save results
    res_dir = ROOT / "results" / "explainability"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    with open(res_dir / "chronos_feature_importance.json", "w") as f:
        json.dump(importances, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    df_imp = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title("Chronos-T5 Foundation Expert: Feature Importance")
    plt.tight_layout()
    plt.savefig(res_dir / "feature_importance_plot.png")
    
    print(f"\n✅ Explainability results saved to {res_dir}")
    print("\nTop 3 Influential Features for Chronos-T5:")
    for i, row in df_imp.head(3).iterrows():
        print(f"- {row['Feature']}: {row['Importance']:.6f}")

if __name__ == "__main__":
    main()
