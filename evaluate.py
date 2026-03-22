"""
evaluate.py
===========
Loads XGBoost out-of-sample predictions and computes
final metrics per ticker and averaged across all tickers.

Usage:
    python evaluate.py
"""

import pandas as pd
import numpy as np
import os


def compute_metrics(df_ticker):
    y_true = df_ticker["y_true"].values
    y_pred = df_ticker["y_pred"].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    rolling_std = pd.Series(y_pred).rolling(20).std().fillna(
        pd.Series(y_pred).std()
    ).values
    position = np.clip(y_pred / (rolling_std + 1e-8), -2, 2)
    strategy_return = y_true * position
    sharpe = (np.mean(strategy_return) /
              (np.std(strategy_return) + 1e-8)) * np.sqrt(252)

    return rmse, mae, da, sharpe


def main():
    path = "results/xgboost_predictions.csv"
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run models/baseline_xgboost.py first.")
        return

    df = pd.read_csv(path)
    tickers = df["Ticker"].unique()

    print(f"\n{'Ticker':<15} {'RMSE':>8} {'MAE':>8} {'DA%':>8} {'Sharpe':>8}")
    print("-" * 55)

    all_metrics = []
    for ticker in tickers:
        subset = df[df["Ticker"] == ticker]
        rmse, mae, da, sharpe = compute_metrics(subset)
        all_metrics.append([rmse, mae, da, sharpe])
        print(f"{ticker:<15} {rmse:>8.4f} {mae:>8.4f} {da:>8.2f} {sharpe:>8.2f}")

    avg = np.mean(all_metrics, axis=0)
    print("-" * 55)
    print(f"{'Average':<15} {avg[0]:>8.4f} {avg[1]:>8.4f} "
          f"{avg[2]:>8.2f} {avg[3]:>8.2f}")

    out = "results/baseline_results.csv"
    rows = []
    for ticker, m in zip(tickers, all_metrics):
        rows.append({"Ticker": ticker, "RMSE": m[0], "MAE": m[1],
                     "DA%": m[2], "Sharpe": m[3]})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
