"""
RAMT Baseline Model — PyTorch LSTM
Walk-forward validation with expanding training window (same schedule as baseline_xgboost).
Predicts next-day log return from sequences of 30 trading days.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data" / "processed").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_FILES = [
    ("JPM", "JPM_features.csv"),
    ("RELIANCE_NS", "RELIANCE_NS_features.csv"),
    ("TCS_NS", "TCS_NS_features.csv"),
    ("HDFCBANK_NS", "HDFCBANK_NS_features.csv"),
    ("EPIGRAL_NS", "EPIGRAL_NS_features.csv"),
]

EXCLUDE_FROM_X = {
    "Date",
    "Log_Return",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Ticker",
    "HMM_Regime_Label",
}

INITIAL_TRAIN_FRAC = 0.6
STEP_DAYS = 63
TEST_DAYS = 63
SEQ_LEN = 30
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
VAL_FRAC_OF_SEQUENCES = 0.2
LEARNING_RATE = 0.001


def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """
    Build aligned features and next-day log-return target.

    Returns:
        X: Numeric feature matrix (no raw OHLCV, no target leakage).
        y: Next-day log return aligned to each row.
        dates: Row dates (same length as X).
        regime: HMM_Regime per row (int; -1 if missing), for parity with other baselines.
        numeric_cols: Column names used in X (same order as baseline feature groups).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    y = df["Log_Return"].shift(-1)
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    dates = df["Date"]
    regime = df["HMM_Regime"].fillna(-1).astype(int)
    candidate_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    numeric_cols = [
        c
        for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[numeric_cols].astype(np.float32)
    return X, y, dates, regime, numeric_cols


def make_sequence_tensors(
    X_scaled: np.ndarray,
    y: np.ndarray,
    train_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build all training sequences whose last timestep index is in [0, train_end).

    Each sample uses rows [end_idx - SEQ_LEN + 1, ..., end_idx] and target y[end_idx].
    """
    starts = np.arange(0, train_end - SEQ_LEN + 1, dtype=np.int64)
    ends = starts + SEQ_LEN - 1
    n = len(starts)
    n_feat = X_scaled.shape[1]
    X_seq = np.zeros((n, SEQ_LEN, n_feat), dtype=np.float32)
    y_seq = np.zeros((n, 1), dtype=np.float32)
    for i, e in enumerate(ends):
        s = e - SEQ_LEN + 1
        X_seq[i] = X_scaled[s : e + 1]
        y_seq[i, 0] = float(y[e])
    return torch.from_numpy(X_seq), torch.from_numpy(y_seq)


def split_train_val_sequences(
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
    val_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split sequence tensors chronologically: first (1 - val_frac) for train,
    last val_frac for validation.
    """
    n = X_seq.shape[0]
    n_val = max(1, int(np.ceil(val_frac * n)))
    n_fit = n - n_val
    if n_fit < 1:
        raise ValueError("Not enough sequences for train/val split")
    return (
        X_seq[:n_fit],
        y_seq[:n_fit],
        X_seq[n_fit:],
        y_seq[n_fit:],
    )


def make_test_sequence_tensors(
    X_scaled_prefix: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    test_start: int,
    test_end: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build test sequences for indices test_start .. test_end - 1 (inclusive).

    Requires a scaled feature matrix for rows [0, test_end) so each window has history.
    Returns X tensor, y tensor, and datetime64 dates aligned to each sequence end.
    """
    n_test = test_end - test_start
    n_feat = X_scaled_prefix.shape[1]
    X_seq = np.zeros((n_test, SEQ_LEN, n_feat), dtype=np.float32)
    y_seq = np.zeros((n_test, 1), dtype=np.float32)
    date_list: list[np.datetime64] = []
    for k, end_idx in enumerate(range(test_start, test_end)):
        s = end_idx - SEQ_LEN + 1
        X_seq[k] = X_scaled_prefix[s : end_idx + 1]
        y_seq[k, 0] = float(y[end_idx])
        date_list.append(np.datetime64(dates.iloc[end_idx]))
    return (
        torch.from_numpy(X_seq),
        torch.from_numpy(y_seq),
        np.array(date_list, dtype="datetime64[ns]"),
    )


class LSTMModel(nn.Module):
    """
    Two-layer stacked LSTM for sequence regression: (batch, seq_len, n_features) -> (batch, 1).
    """

    def __init__(self, input_size: int) -> None:
        """Initialize LSTM stacks and output head."""
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run LSTMs, take last timestep, dropout, and linear readout."""
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
) -> float:
    """
    Train with Adam and MSE; early-stop on validation MSE.

    Returns the best validation loss achieved.
    """
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    bad_epochs = 0

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            target = yb.squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).squeeze(-1)
                target = yb.squeeze(-1)
                val_loss_sum += float(criterion(pred, target).item())
                val_batches += 1
        val_mse = val_loss_sum / max(val_batches, 1)

        if val_mse < best_val - 1e-9:
            best_val = val_mse
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return float(best_val)


def predict_batches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return predictions as a 1D float array."""
    model.eval()
    preds: list[float] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb).squeeze(-1).cpu().numpy()
            preds.extend(out.tolist())
    return np.asarray(preds, dtype=float)


def walk_forward_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    ticker: str,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expanding-window walk-forward: same initial fraction and step as baseline_xgboost.

    Per fold: fit StandardScaler on training rows only, build sequences, train a fresh
    LSTM with early stopping (last 20% of train sequences = validation), predict test fold.

    Each sequence uses the prior SEQ_LEN days of features; the target is next-day log return
    at the sequence end; HMM_Regime at the last timestep is included in the feature columns.
    """
    n = len(X)
    n0 = int(n * INITIAL_TRAIN_FRAC)
    if n0 < SEQ_LEN or n0 + TEST_DAYS > n:
        raise ValueError(f"{ticker}: insufficient rows (n={n}, n0={n0}, seq_len={SEQ_LEN})")

    train_end_probe = n0
    total_folds = 0
    while train_end_probe + TEST_DAYS <= n:
        total_folds += 1
        train_end_probe += TEST_DAYS

    print(f"=== WALK-FORWARD {ticker} ===")
    print(f"Initial train size: {n0}")
    print(f"Seq len: {SEQ_LEN}")
    print(f"Total folds: {total_folds}")

    X_np = X.to_numpy(dtype=np.float32)
    y_np = y.to_numpy(dtype=np.float32)

    oos_dates: list[np.datetime64] = []
    oos_y_true: list[float] = []
    oos_y_pred: list[float] = []

    train_end = n0
    fold_idx = 0
    while train_end + TEST_DAYS <= n:
        fold_idx += 1
        test_start = train_end
        test_end = train_end + TEST_DAYS

        X_tr = X_np[:train_end]
        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_train_scaled = scaler.transform(X_tr)
        X_prefix_scaled = scaler.transform(X_np[:test_end])

        X_seq_all, y_seq_all = make_sequence_tensors(X_train_scaled, y_np, train_end)
        X_tr_fit, y_tr_fit, X_val, y_val = split_train_val_sequences(
            X_seq_all, y_seq_all, VAL_FRAC_OF_SEQUENCES
        )

        train_ds = TensorDataset(X_tr_fit, y_tr_fit)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        n_in = len(feature_cols)
        model = LSTMModel(n_in).to(device)
        best_val = train_with_early_stopping(
            model,
            train_loader,
            val_loader,
            device,
            MAX_EPOCHS,
            EARLY_STOP_PATIENCE,
            LEARNING_RATE,
        )

        X_test, y_test, fold_dates = make_test_sequence_tensors(
            X_prefix_scaled,
            y_np,
            dates,
            test_start,
            test_end,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        y_pred_fold = predict_batches(model, test_loader, device)
        y_true_fold = y_test.squeeze(-1).numpy()

        print(
            f"Fold {fold_idx}/{total_folds}: train_size={train_end} "
            f"test_size={TEST_DAYS} best_val_loss={best_val:.6f}"
        )

        oos_dates.extend(list(fold_dates))
        oos_y_true.extend(y_true_fold.tolist())
        oos_y_pred.extend(y_pred_fold.tolist())

        train_end += TEST_DAYS

    return (
        np.array(oos_dates, dtype="datetime64[ns]"),
        np.asarray(oos_y_true, dtype=float),
        np.asarray(oos_y_pred, dtype=float),
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error between two 1D arrays."""
    return float(np.sqrt(mean_squared_error(a, b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two 1D arrays."""
    return float(mean_absolute_error(a, b))


def directional_accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of times predicted and true returns have the same sign."""
    st = np.sign(y_true)
    sp = np.sign(y_pred)
    return float(np.mean(st == sp) * 100.0)


def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Sharpe-style metric matching baseline_xgboost: soft position from scaled predictions.
    """
    y_pred_s = pd.Series(y_pred, dtype=float)
    roll_std = y_pred_s.rolling(20, min_periods=1).std()
    pos = y_pred_s / roll_std.replace(0.0, np.nan)
    pos = pos.clip(-2.0, 2.0).fillna(0.0)
    strat = y_true * pos.to_numpy()
    mu = float(np.mean(strat))
    sig = float(np.std(strat, ddof=0))
    if sig == 0.0 or not np.isfinite(sig):
        return float("nan")
    return mu / sig * np.sqrt(252.0)


def main() -> None:
    """Load data, run walk-forward LSTM per ticker, save predictions and print metrics."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_raw: dict[str, pd.DataFrame] = {}
    for ticker, fname in FEATURE_FILES:
        path = PROCESSED_DIR / fname
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Loading from: {os.path.abspath(path)}")
        data_raw[ticker] = pd.read_csv(path, parse_dates=["Date"])

    print("=== DATA LOADED ===")
    for ticker, df in data_raw.items():
        print(
            f"{ticker}: {len(df)} rows, date range: {df['Date'].min()} → {df['Date'].max()}"
        )

    first_ticker = FEATURE_FILES[0][0]
    _, _, _, _, feat0 = prepare_xy(data_raw[first_ticker])
    template = LSTMModel(len(feat0))
    n_params = sum(p.numel() for p in template.parameters())
    print(f"LSTM parameter count: {n_params}")

    all_rows: list[dict] = []
    metrics_rows: list[tuple[str, float, float, float, float]] = []

    for ticker, fname in FEATURE_FILES:
        raw = data_raw[ticker]
        X, y, dates, _regime, feature_cols = prepare_xy(raw)
        print(f"{ticker}: using {len(feature_cols)} numeric features")

        dates_oos, y_true, y_pred = walk_forward_lstm(
            X, y, dates, ticker, feature_cols, device
        )

        r = rmse(y_true, y_pred)
        m = mae(y_true, y_pred)
        da = directional_accuracy_pct(y_true, y_pred)
        sh = sharpe_ratio(y_true, y_pred)
        metrics_rows.append((ticker, r, m, da, sh))

        for d, yt, yp in zip(dates_oos, y_true, y_pred):
            all_rows.append(
                {
                    "Date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                }
            )

    out_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "lstm_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print()
    print(f"Saved predictions → {out_path.resolve()}")
    print()
    print("Table — LSTM Walk-Forward Results:")
    print("Ticker       | RMSE   | MAE    | DA%   | Sharpe")
    print("-------------|--------|--------|-------|-------")

    rmses, maes, das, shs = [], [], [], []
    for tkr, r, m, da, sh in metrics_rows:
        rmses.append(r)
        maes.append(m)
        das.append(da)
        shs.append(sh if not np.isnan(sh) else np.nan)
        sh_str = f"{sh:>6.2f}" if not np.isnan(sh) else "   nan"
        print(f"{tkr:<12} | {r:.4f} | {m:.4f} | {da:>5.2f} | {sh_str}")

    avg_r = float(np.nanmean(rmses))
    avg_m = float(np.nanmean(maes))
    avg_da = float(np.nanmean(das))
    finite_sh = [s for s in shs if not np.isnan(s)]
    avg_sh = float(np.nanmean(finite_sh)) if finite_sh else float("nan")
    if not finite_sh:
        avg_sh_str = "   nan"
    else:
        avg_sh_str = f"{avg_sh:>6.2f}"
    print(f"{'Average':<12} | {avg_r:.4f} | {avg_m:.4f} | {avg_da:>5.2f} | {avg_sh_str}")
    print()


if __name__ == "__main__":
    main()
