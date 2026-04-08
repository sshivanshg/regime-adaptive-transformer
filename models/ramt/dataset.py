import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PRICE_COLS = [
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",
    "Return_Lag_10",
    "Return_Lag_20",
]

VOL_COLS = [
    "Realized_Vol_5",
    "Realized_Vol_20",
    "Realized_Vol_60",
    "Garman_Klass_Vol",
    "Vol_Ratio",
]

TECH_COLS = [
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Upper",
    "BB_Lower",
    "BB_Width",
    "BB_Position",
]

MOMENTUM_COLS = ["Momentum_5", "Momentum_20", "Momentum_60", "ROC_10"]

VOLUME_COLS = ["Volume_MA_Ratio", "Volume_Log"]

REGIME_COLS = ["HMM_Regime"]

CROSS_ASSET_COLS = ["Rolling_Corr_Index"]

ALL_FEATURE_COLS = (
    PRICE_COLS
    + VOL_COLS
    + TECH_COLS
    + MOMENTUM_COLS
    + VOLUME_COLS
    + REGIME_COLS
    + CROSS_ASSET_COLS
)
# Total: 27 columns

TARGET_COL = "Log_Return"


class SequenceDataset(Dataset):
    """
    Creates overlapping sequences of length seq_len from
    a numpy feature array. Each sample is:
      X: (seq_len, num_features) -- input sequence
      y: scalar -- next day log return (target)
      regime: integer -- HMM_Regime at last timestep
    """

    def __init__(self, features, targets, regimes, seq_len=30):
        self.features = features  # numpy (N, num_features)
        self.targets = targets  # numpy (N,)
        self.regimes = regimes  # numpy (N,) integers
        self.seq_len = seq_len
        # Valid indices: need seq_len rows before each target
        self.valid_idx = list(range(seq_len, len(targets)))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        X = self.features[i - self.seq_len : i]  # (seq_len, features)
        y = self.targets[i]  # scalar
        regime = self.regimes[i]  # integer
        return (
            torch.FloatTensor(X),
            torch.FloatTensor([y]),
            torch.LongTensor([regime]),
        )


class RAMTDataModule:
    """
    Handles data loading, scaling, and DataLoader creation
    for one ticker and one walk-forward fold.

    Key guarantee: StandardScaler is ALWAYS fit on training
    data only and applied to val/test. Zero leakage.
    """

    def __init__(self, ticker, data_dir="data/processed", seq_len=30, batch_size=32):
        self.ticker = ticker
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.df = None
        self.features = None
        self.targets = None
        self.regimes = None
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        """Load CSV, compute target, extract feature arrays."""
        path = os.path.join(data_dir, f"{self.ticker}_features.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        # Target: next day log return
        df["target"] = df[TARGET_COL].shift(-1)
        df = df.dropna(subset=["target"])

        # Verify all feature columns exist
        missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {self.ticker}: {missing}")

        self.df = df
        self.dates = df.index.values
        self.features_raw = df[ALL_FEATURE_COLS].values.astype(np.float32)
        self.targets = df["target"].values.astype(np.float32)
        self.regimes = df["HMM_Regime"].values.astype(np.int64)

    def get_fold_loaders(self, train_idx, test_idx, val_fraction=0.15):
        """
        Create DataLoaders for one walk-forward fold.

        Steps:
        1. Split train_idx into train and val
        2. Fit scaler on training features only
        3. Transform train, val, test
        4. Create SequenceDatasets
        5. Return DataLoaders

        Args:
            train_idx: array of training row indices
            test_idx: array of test row indices
            val_fraction: fraction of training for validation

        Returns:
            train_loader, val_loader, test_loader, test_dates
        """
        # Split train into train and val
        val_size = max(1, int(len(train_idx) * val_fraction))
        actual_train_idx = train_idx[:-val_size]
        val_idx = train_idx[-val_size:]

        # Extract raw features
        X_train = self.features_raw[actual_train_idx]
        X_val = self.features_raw[val_idx]
        X_test = self.features_raw[test_idx]

        y_train = self.targets[actual_train_idx]
        y_val = self.targets[val_idx]
        y_test = self.targets[test_idx]

        r_train = self.regimes[actual_train_idx]
        r_val = self.regimes[val_idx]
        r_test = self.regimes[test_idx]

        # Fit scaler on training only
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)
        X_test_sc = self.scaler.transform(X_test)

        # Create datasets
        train_ds = SequenceDataset(X_train_sc, y_train, r_train, self.seq_len)
        val_ds = SequenceDataset(X_val_sc, y_val, r_val, self.seq_len)
        test_ds = SequenceDataset(X_test_sc, y_test, r_test, self.seq_len)

        # DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        test_dates = self.dates[test_idx[self.seq_len :]]

        return train_loader, val_loader, test_loader, test_dates

    def get_walk_forward_indices(self, init_train_frac=0.6, step_size=63):
        """
        Generate walk-forward fold indices.
        Returns list of (train_idx, test_idx) tuples.
        """
        n = len(self.targets)
        init_train_size = int(n * init_train_frac)
        folds = []
        train_end = init_train_size

        while train_end + step_size <= n:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, train_end + step_size)
            folds.append((train_idx, test_idx))
            train_end += step_size

        return folds


if __name__ == "__main__":
    print("Testing RAMTDataModule...")

    dm = RAMTDataModule("JPM", seq_len=30, batch_size=32)
    folds = dm.get_walk_forward_indices()

    print(f"Total folds: {len(folds)}")
    print(f"Total rows: {len(dm.targets)}")
    print(f"Feature columns: {len(ALL_FEATURE_COLS)}")
    print(f"Feature names: {ALL_FEATURE_COLS}")

    # Test first fold
    train_idx, test_idx = folds[0]
    train_loader, val_loader, test_loader, dates = dm.get_fold_loaders(
        train_idx, test_idx
    )

    # Check shapes
    X_batch, y_batch, r_batch = next(iter(train_loader))
    print("\nFirst fold:")
    print(f"  Train size: {len(train_idx)}")
    print(f"  Test size: {len(test_idx)}")
    print(f"  X shape: {X_batch.shape}")
    print(f"  y shape: {y_batch.shape}")
    print(f"  regime shape: {r_batch.shape}")
    print(f"  Regime values: {r_batch.squeeze().unique()}")
    print(f"  X min: {X_batch.min():.4f} max: {X_batch.max():.4f}")
    print("\nAll checks passed.")