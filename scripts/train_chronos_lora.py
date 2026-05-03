from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.lora_experiment.chronos_lora_v2 import ChronosLoRARankerV2
from models.ramt.dataset import ALL_FEATURE_COLS, build_ticker_universe

# --- Configuration ---
SEQ_LEN = 30
TARGET_COL = "Sector_Alpha"
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-15")

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

class TickerSequenceDataset(Dataset):
    def __init__(self, ticker_data: dict, sample_index: list, seq_len: int):
        self.ticker_data = ticker_data
        self.sample_index = sample_index
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        ticker, i = self.sample_index[idx]
        data = self.ticker_data[ticker]
        
        x = data['features'][i - self.seq_len : i]
        y = data['target'][i]
        
        return torch.from_numpy(x).float(), torch.tensor(y).float()

def _collect_data(processed_dir: Path, feature_cols: list):
    tickers = build_ticker_universe(str(processed_dir))
    ticker_data = {}
    train_index = []
    test_index = []

    print(f"Loading data for {len(tickers)} tickers...")
    for ticker in tqdm(tickers):
        p = processed_dir / f"{ticker}_features.parquet"
        if not p.exists(): continue
        
        df = pd.read_parquet(p)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=[*feature_cols, TARGET_COL])
        
        if len(df) <= SEQ_LEN: continue
        
        feats = df[feature_cols].to_numpy(dtype=np.float32)
        target = df[TARGET_COL].to_numpy(dtype=np.float32)
        dates = df["Date"].tolist()
        
        ticker_data[ticker] = {'features': feats, 'target': target, 'dates': dates}
        
        for i in range(SEQ_LEN, len(df)):
            dt = dates[i]
            if dt <= TRAIN_END:
                train_index.append((ticker, i))
            elif dt >= TEST_START and dt <= TEST_END:
                test_index.append((ticker, i))
                
    return ticker_data, train_index, test_index

def train():
    processed_dir = ROOT / "data" / "processed"
    out_dir = ROOT / "models" / "lora_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    feature_cols = list(ALL_FEATURE_COLS)
    ticker_data, train_idx, test_idx = _collect_data(processed_dir, feature_cols)
    
    train_ds = TickerSequenceDataset(ticker_data, train_idx, SEQ_LEN)
    test_ds = TickerSequenceDataset(ticker_data, test_idx, SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    
    model = ChronosLoRARankerV2(input_dim=len(feature_cols)).to(DEVICE)
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
    # Save the model
    save_path = out_dir / "chronos_v2_adapter.pt"
    torch.save({
        'encoder_lora_state_dict': model.encoder.state_dict(),
        'input_projection_state_dict': model.input_projection.state_dict(),
        'ranking_head_state_dict': model.ranking_head.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")
    
    # Evaluation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            preds.extend(pred)
            actuals.extend(y.numpy())
            
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    da = np.mean(np.sign(preds) == np.sign(actuals))
    # IC (Spearman Correlation)
    from scipy.stats import spearmanr
    ic, _ = spearmanr(preds, actuals)
    
    metrics = {
        "directional_accuracy": float(da),
        "information_coefficient": float(ic),
        "test_mse": float(np.mean((preds - actuals)**2))
    }
    
    metrics_path = ROOT / "results" / "lora" / "lora_v2_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    print(f"Final Metrics: {metrics}")

if __name__ == "__main__":
    train()
