#!/usr/bin/env python3
"""
Train RAMT, LSTM, or XGBoost on windowed OHLCV features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.feature_engineering import FeatureConfig, build_dataset_from_csv
from models.baseline_lstm import init_model as init_lstm
from models.baseline_xgboost import last_timestep, train_xgboost
from models.ramt import RegimeAdaptiveTransformer, bce_with_logits_loss


def _load_split(ds: dict) -> tuple:
    X, y = ds["X"], ds["y"]
    tr, va, te = ds["train_idx"], ds["val_idx"], ds["test_idx"]
    return (
        X[tr],
        y[tr],
        X[va],
        y[va],
        X[te],
        y[te],
        X.shape[2],
    )


def train_torch(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    loss_fn,
) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
    )
    va_x = torch.from_numpy(X_va).to(device)
    va_y = torch.from_numpy(y_va).to(device)
    loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if isinstance(model, RegimeAdaptiveTransformer):
                logits, _, aux = model(xb)
                loss = loss_fn(logits, yb, aux)
            else:
                logits = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            if isinstance(model, RegimeAdaptiveTransformer):
                lv, _, _ = model(va_x)
                vloss = nn.functional.binary_cross_entropy_with_logits(
                    lv, va_y.float()
                ).item()
            else:
                lv = model(va_x)
                vloss = nn.functional.binary_cross_entropy_with_logits(
                    lv, va_y.float()
                ).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {ep+1}/{epochs} train_loss={np.mean(losses):.4f} val_loss={vloss:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ramt", "lstm", "xgboost"], default="ramt")
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "SPY.csv",
        help="OHLCV CSV from data/download.py",
    )
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "checkpoints",
    )
    args = p.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Run: python data/download.py --tickers SPY"
        )

    cfg = FeatureConfig(window=args.window)
    ds = build_dataset_from_csv(args.data, cfg)
    X_tr, y_tr, X_va, y_va, X_te, y_te, fdim = _load_split(ds)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.model == "xgboost":
        clf = train_xgboost(
            last_timestep(X_tr),
            y_tr,
            last_timestep(X_va),
            y_va,
        )
        path = args.out / "xgboost.joblib"
        joblib.dump({"model": clf, "feature_config": cfg}, path)
        print(f"Saved {path}")
        return

    if args.model == "lstm":
        m = init_lstm(fdim, hidden=64, num_layers=2)
        m = train_torch(
            m,
            X_tr,
            y_tr,
            X_va,
            y_va,
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            None,
        )
        ckpt = {
            "state_dict": m.state_dict(),
            "feature_config": cfg,
            "input_dim": fdim,
            "kind": "lstm",
        }
        path = args.out / "best.pt"
        torch.save(ckpt, path)
        print(f"Saved {path}")
        return

    m = RegimeAdaptiveTransformer(
        input_dim=fdim,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        num_experts=4,
        expert_hidden=128,
    )
    m = train_torch(
        m,
        X_tr,
        y_tr,
        X_va,
        y_va,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
        bce_with_logits_loss,
    )
    ckpt = {
        "state_dict": m.state_dict(),
        "feature_config": cfg,
        "input_dim": fdim,
        "kind": "ramt",
    }
    path = args.out / "best.pt"
    torch.save(ckpt, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
