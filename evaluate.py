#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the chronological test split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.feature_engineering import build_dataset_from_csv
from models.baseline_lstm import init_model as init_lstm
from models.baseline_xgboost import evaluate_classifier, last_timestep
from models.ramt import RegimeAdaptiveTransformer


def report_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(np.int64)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    else:
        out["roc_auc"] = float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "SPY.csv",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "best.pt",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Missing {args.data}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing {args.checkpoint}")

    if args.checkpoint.suffix in (".joblib", ".pkl", ".pickle"):
        blob = joblib.load(args.checkpoint)
    else:
        try:
            blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(args.checkpoint, map_location="cpu")

    if isinstance(blob, dict) and "model" in blob and hasattr(blob["model"], "predict_proba"):
        cfg = blob.get("feature_config")
        if cfg is None:
            raise SystemExit("Checkpoint missing feature_config")
        ds = build_dataset_from_csv(args.data, cfg)
        X_te = last_timestep(ds["X"][ds["test_idx"]])
        y_te = ds["y"][ds["test_idx"]]
        ev = evaluate_classifier(blob["model"], X_te, y_te)
        print(ev)
        return

    cfg = blob["feature_config"]
    ds = build_dataset_from_csv(args.data, cfg)
    X_te = ds["X"][ds["test_idx"]]
    y_te = ds["y"][ds["test_idx"]]
    device = torch.device(args.device)

    kind = blob.get("kind", "ramt")
    fdim = int(blob["input_dim"])

    if kind == "lstm":
        m = init_lstm(fdim)
    else:
        m = RegimeAdaptiveTransformer(input_dim=fdim)

    m.load_state_dict(blob["state_dict"])
    m.to(device)
    m.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_te).to(device)
        if isinstance(m, RegimeAdaptiveTransformer):
            logits, _, _ = m(x)
        else:
            logits = m(x)
        proba = torch.sigmoid(logits).cpu().numpy()
    print(report_metrics(y_te, proba))


if __name__ == "__main__":
    main()
