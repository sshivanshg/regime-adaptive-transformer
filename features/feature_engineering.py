"""
Build supervised learning matrices: rolling windows, features, binary next-return labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    window: int = 32
    neutral_band_bp: float = 0.0  # drop |r| below this (in basis points) if > 0
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        df = df.drop(columns=["datetime"])
    df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df
    return df


def _log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row features before windowing (aligned to same index as df)."""
    c = df["close"]
    r = _log_return(c)
    h = df["high"]
    l = df["low"]
    o = df["open"]
    hl = np.log(h / l).replace([np.inf, -np.inf], np.nan)
    co = np.log(c / o).replace([np.inf, -np.inf], np.nan)
    vol = r.rolling(20, min_periods=5).std()
    mom = c.pct_change(10)
    out = pd.DataFrame(
        {
            "ret1": r,
            "hl_range": hl,
            "co": co,
            "vol20": vol,
            "mom10": mom,
        },
        index=df.index,
    )
    return out.dropna()


def make_windows(
    feat: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: (N, T, F) float32
    y: (N,) int64 in {0,1}
    valid_idx: index in feat of window end (for alignment)
    """
    arr = feat.to_numpy(dtype=np.float64)
    n, fdim = arr.shape
    w = cfg.window
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ends: list[int] = []

    ret1 = feat["ret1"].to_numpy()
    for end in range(w, n - 1):
        next_r = ret1[end + 1]
        if cfg.neutral_band_bp > 0:
            thr = cfg.neutral_band_bp / 1e4
            if abs(next_r) < thr:
                continue
        xs.append(arr[end - w + 1 : end + 1])
        ys.append(1 if next_r > 0 else 0)
        ends.append(end)

    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, np.asarray(ends, dtype=np.int64)


def time_series_split_indices(
    n: int,
    test_size: float,
    val_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological: train | val | test."""
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough samples for train/val/test split")
    idx = np.arange(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def scale_features_per_split(
    X: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on all timesteps in train split (flatten T*F).
    """
    t, f = X.shape[1], X.shape[2]
    scaler = StandardScaler()
    tr = X[train_idx].reshape(-1, f)
    scaler.fit(tr)
    flat = X.reshape(-1, f)
    X_scaled = scaler.transform(flat).reshape(X.shape[0], t, f).astype(np.float32)
    return X_scaled, scaler


def build_dataset_from_csv(path: Path, cfg: FeatureConfig | None = None) -> dict:
    cfg = cfg or FeatureConfig()
    df = load_ohlcv_csv(path)
    feat = build_feature_frame(df)
    feat = feat.reset_index(drop=True)
    X, y, ends = make_windows(feat, cfg)
    n = len(y)
    train_idx, val_idx, test_idx = time_series_split_indices(
        n, cfg.test_size, cfg.val_size
    )
    X_scaled, scaler = scale_features_per_split(X, train_idx)
    return {
        "X": X_scaled,
        "y": y,
        "ends": ends,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "scaler": scaler,
        "feature_names": list(feat.columns),
        "config": cfg,
    }
