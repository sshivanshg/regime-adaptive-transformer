from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from models.lora_experiment.chronos_lora import ChronosLoRARanker
from models.ramt.dataset import ALL_FEATURE_COLS, build_ticker_universe

SEQ_LEN = 30
TARGET_COL = "Sector_Alpha"
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-10")
TEST_END = pd.Timestamp.max

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


@dataclass
class TickerPanelData:
    features: np.ndarray
    target: np.ndarray
    dates: np.ndarray


@dataclass
class SplitIndex:
    ticker_data: dict[str, TickerPanelData]
    train_index: list[tuple[str, int]]
    test_index: list[tuple[str, int]]


class IndexedSequenceDataset(Dataset):
    def __init__(
        self,
        ticker_data: dict[str, TickerPanelData],
        sample_index: list[tuple[str, int]],
        seq_len: int,
        include_meta: bool = False,
    ) -> None:
        self.ticker_data = ticker_data
        self.sample_index = sample_index
        self.seq_len = int(seq_len)
        self.include_meta = bool(include_meta)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        ticker, i = self.sample_index[idx]
        panel = self.ticker_data[ticker]
        x_i = torch.from_numpy(panel.features[i - self.seq_len : i])
        y_i = torch.tensor(float(panel.target[i]), dtype=torch.float32)

        if not self.include_meta:
            return x_i, y_i

        d_str = str(pd.Timestamp(panel.dates[i]).date())
        return x_i, y_i, d_str, ticker


def _resolved_target_column(df: pd.DataFrame) -> str | None:
    if TARGET_COL in df.columns and df[TARGET_COL].notna().any():
        return TARGET_COL
    if "Monthly_Alpha" in df.columns and df["Monthly_Alpha"].notna().any():
        return "Monthly_Alpha"
    return None


def _iter_processed_tickers(processed_dir: Path) -> Iterable[str]:
    for ticker in build_ticker_universe(str(processed_dir)):
        if (processed_dir / f"{ticker}_features.parquet").is_file():
            yield ticker


def _collect_split_index(processed_dir: Path, feature_cols: list[str], seq_len: int) -> SplitIndex:
    ticker_data: dict[str, TickerPanelData] = {}
    train_index: list[tuple[str, int]] = []
    test_index: list[tuple[str, int]] = []

    needed_cols = ["Date", *feature_cols, TARGET_COL, "Monthly_Alpha"]
    for ticker in _iter_processed_tickers(processed_dir):
        df = pd.read_parquet(processed_dir / f"{ticker}_features.parquet")
        existing_cols = [c for c in needed_cols if c in df.columns]
        df = df[existing_cols].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        target_col = _resolved_target_column(df)
        if target_col is None:
            continue

        df = df.dropna(subset=[*feature_cols, target_col])
        if len(df) <= seq_len:
            continue

        feats = df[feature_cols].to_numpy(dtype=np.float32)
        target = df[target_col].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(df["Date"]).to_numpy()
        ticker_data[ticker] = TickerPanelData(features=feats, target=target, dates=dates)

        for i in range(seq_len, len(df)):
            d = pd.Timestamp(dates[i])
            if d <= TRAIN_END:
                train_index.append((ticker, i))
            elif d >= TEST_START and d <= TEST_END:
                test_index.append((ticker, i))

    if not train_index or not test_index:
        raise RuntimeError(
            "No train/test samples built from parquet files. Check data coverage and split dates."
        )

    return SplitIndex(
        ticker_data=ticker_data,
        train_index=train_index,
        test_index=test_index,
    )


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.sign(pred) == np.sign(actual)))


def _mean_ic(df: pd.DataFrame) -> float:
    ics: list[float] = []
    for _, grp in df.groupby("Date"):
        if len(grp) < 4:
            continue
        ic = grp["predicted"].corr(grp["actual"], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else float("nan")


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def _extract_metric(payload: dict, *keys: str) -> float | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except Exception:
                pass
    return None


def _print_comparison_table(ramt_metrics_path: Path, lora_metrics_path: Path) -> None:
    if not lora_metrics_path.is_file():
        print(f"LoRA metrics file missing: {lora_metrics_path}")
        return

    with lora_metrics_path.open("r", encoding="utf-8") as f:
        lora = json.load(f)

    if ramt_metrics_path.is_file():
        with ramt_metrics_path.open("r", encoding="utf-8") as f:
            ramt = json.load(f)
    else:
        print(f"RAMT metrics file missing: {ramt_metrics_path}")
        ramt = {}

    ramt_da = _extract_metric(ramt, "directional_accuracy")
    if ramt_da is None:
        da_pct = _extract_metric(ramt, "DA_pct")
        ramt_da = (da_pct / 100.0) if da_pct is not None else None

    ramt_mean_ic = _extract_metric(ramt, "mean_ic", "mean_IC")
    ramt_pred_std = _extract_metric(ramt, "prediction_std")
    ramt_rmse = _extract_metric(ramt, "rmse", "RMSE")
    ramt_mae = _extract_metric(ramt, "mae", "MAE")

    lora_da = _extract_metric(lora, "directional_accuracy")
    lora_mean_ic = _extract_metric(lora, "mean_ic")
    lora_pred_std = _extract_metric(lora, "prediction_std")
    lora_rmse = _extract_metric(lora, "rmse")
    lora_mae = _extract_metric(lora, "mae")

    def fmt(v: float | None) -> str:
        if v is None or not np.isfinite(v):
            return "n/a"
        return f"{v:.6f}"

    print("\nRAMT vs LoRA metrics")
    print("Metric                | RAMT      | LoRA")
    print("----------------------|-----------|-----------")
    print(f"directional_accuracy  | {fmt(ramt_da):>9} | {fmt(lora_da):>9}")
    print(f"mean_ic               | {fmt(ramt_mean_ic):>9} | {fmt(lora_mean_ic):>9}")
    print(f"prediction_std        | {fmt(ramt_pred_std):>9} | {fmt(lora_pred_std):>9}")
    print(f"rmse                  | {fmt(ramt_rmse):>9} | {fmt(lora_rmse):>9}")
    print(f"mae                   | {fmt(ramt_mae):>9} | {fmt(lora_mae):>9}")

    if lora_pred_std is None or not np.isfinite(lora_pred_std):
        print("\nCollapse check (prediction_std > 0.01): unavailable")
    else:
        status = "PASS" if lora_pred_std > 0.01 else "FAIL"
        print(f"\nCollapse check (prediction_std > 0.01): {status} ({lora_pred_std:.6f})")


def train_and_evaluate() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    feature_cols = list(ALL_FEATURE_COLS)
    if len(feature_cols) != 10:
        raise ValueError(
            f"Expected 10 RAMT features, found {len(feature_cols)} from ALL_FEATURE_COLS: {feature_cols}"
        )

    root = Path.cwd()
    processed_dir = root / "data" / "processed"
    out_dir = root / "results" / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building train/test arrays from data/processed/*.parquet ...")
    split = _collect_split_index(processed_dir=processed_dir, feature_cols=feature_cols, seq_len=SEQ_LEN)
    print(
        "Single-segment split: "
        f"train<= {TRAIN_END.date()} samples={len(split.train_index)}, "
        f"test>= {TEST_START.date()} samples={len(split.test_index)}"
    )

    train_loader = DataLoader(
        IndexedSequenceDataset(
            ticker_data=split.ticker_data,
            sample_index=split.train_index,
            seq_len=SEQ_LEN,
            include_meta=False,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        IndexedSequenceDataset(
            ticker_data=split.ticker_data,
            sample_index=split.test_index,
            seq_len=SEQ_LEN,
            include_meta=True,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    device = _device()
    print(f"Using device: {device}")

    model = ChronosLoRARanker(
        model_name="amazon/chronos-t5-small",
        input_dim=len(feature_cols),
        lora_r=8,
        lora_alpha=16,
    ).to(device)
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")

    criterion = nn.MSELoss()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_obs = 0
        batch_num = 0
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

            bs = int(y_b.shape[0])
            running_loss += float(loss.item()) * bs
            n_obs += bs
            batch_num += 1

            if batch_num % 50 == 0:
                batch_avg_loss = running_loss / max(n_obs, 1)
                print(f"  Epoch {epoch:02d} - batch {batch_num:4d} - running_mse={batch_avg_loss:.6f} (n={n_obs})")

        epoch_loss = running_loss / max(n_obs, 1)
        print(f"Epoch {epoch:02d}/{EPOCHS} - train_mse={epoch_loss:.6f} (total batches={batch_num})")

    model.eval()
    all_pred: list[float] = []
    all_actual: list[float] = []
    all_dates: list[str] = []
    all_tickers: list[str] = []

    with torch.no_grad():
        for x_b, y_b, dates_b, tickers_b in test_loader:
            x_b = x_b.to(device)
            pred = model(x_b).detach().cpu().numpy().astype(float)
            actual = y_b.detach().cpu().numpy().astype(float)

            all_pred.extend(pred.tolist())
            all_actual.extend(actual.tolist())
            all_dates.extend(list(dates_b))
            all_tickers.extend(list(tickers_b))

    pred_arr = np.asarray(all_pred, dtype=float)
    actual_arr = np.asarray(all_actual, dtype=float)

    pred_df = pd.DataFrame(
        {
            "Date": all_dates,
            "Ticker": all_tickers,
            "predicted": pred_arr,
            "actual": actual_arr,
        }
    ).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    pred_path = out_dir / "lora_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics = {
        "directional_accuracy": _directional_accuracy(pred_arr, actual_arr),
        "mean_ic": _mean_ic(pred_df),
        "prediction_std": float(np.std(pred_arr, ddof=0)),
        "rmse": _rmse(pred_arr, actual_arr),
        "mae": _mae(pred_arr, actual_arr),
    }

    metrics_path = out_dir / "lora_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    adapter_path = out_dir / "chronos_lora_adapter.pt"
    model.save_adapter(adapter_path)

    print(f"Saved adapter: {adapter_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")

    _print_comparison_table(root / "results" / "ramt" / "ramt_metrics.json", metrics_path)


if __name__ == "__main__":
    train_and_evaluate()
