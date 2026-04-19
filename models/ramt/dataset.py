import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Lean NIFTY 200 Parquet feature schema (model inputs only — no regime in X)
# -----------------------------------------------------------------------------

MACRO_COLS = [
    "Macro_INDIAVIX_Ret1d_L1",
    "Macro_CRUDE_Ret1d_L1",
    "Macro_USDINR_Ret1d_L1",
    "Macro_SP500_Ret1d_L1",
]

# Encoder grouping (must partition ALL_FEATURE_COLS without overlap; sum = len(ALL_FEATURE_COLS))
PRICE_COLS = ["Ret_1d", "Ret_5d", "Ret_21d"]
TECH_COLS = ["RSI_14", "BB_Dist"]
VOLUME_COLS = ["Volume_Surge"]
# Legacy names kept empty for imports; regime is NOT in X (passed separately to the model)
VOL_COLS: list[str] = []
MOMENTUM_COLS: list[str] = []
CROSS_ASSET_COLS: list[str] = []
REGIME_COLS: list[str] = []

ALL_FEATURE_COLS = PRICE_COLS + TECH_COLS + VOLUME_COLS + MACRO_COLS

# Separate gate input for the transformer (not part of scaled feature vector)
HMM_REGIME_COL = "HMM_Regime"

# Intra-sector median-demeaned alpha (see features/feature_engineering.apply_sector_alpha_panel).
# Parquets without this column fall back to Monthly_Alpha in LazyTickerStore.
TARGET_COL = "Sector_Alpha"
BENCHMARK_TICKER_STEMS = {"_NSEI", "NSEI", "^NSEI", "NIFTY50", "SP500", "JPM"}
_UNIVERSE_SNAPSHOT_WARNED = False


def _ticker_from_processed_filename(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith("_features"):
        stem = stem[: -len("_features")]
    return stem


def _is_excluded_universe_ticker(stem: str) -> bool:
    return stem.strip().upper() in BENCHMARK_TICKER_STEMS


def build_ticker_universe(processed_dir: str = "data/processed") -> list[str]:
    """
    Build the cross-stock training universe from processed feature files.

    We intentionally keep a static tradable universe snapshot across all dates.
    That avoids turning later index adds/deletes into a time-varying training signal.

    Important: this does NOT solve survivorship bias. If the local files were built
    from a 2026 NIFTY 200 list, the research still uses a 2026 snapshot. The model
    simply does not get to observe membership deletions/additions as a feature over time.
    """
    pdir = Path(processed_dir)
    if not pdir.exists():
        return []

    global _UNIVERSE_SNAPSHOT_WARNED
    if not _UNIVERSE_SNAPSHOT_WARNED:
        warnings.warn(
            "Tradable universe is a static local snapshot (for this repo, typically the "
            "2026 NIFTY 200 list). Historical index membership is not reconstructed, but "
            "the universe is held fixed across all training dates so the model does not "
            "learn add/delete events such as a 2024 removal during 2015 training.",
            RuntimeWarning,
            stacklevel=2,
        )
        _UNIVERSE_SNAPSHOT_WARNED = True

    tickers: list[str] = []
    for p in sorted(list(pdir.glob("*_features.parquet"))):
        t = _ticker_from_processed_filename(p.name)
        if _is_excluded_universe_ticker(t):
            continue
        tickers.append(t)
    return tickers


# Stable ticker universe for embeddings / cross-stock training.
TICKER_LIST = build_ticker_universe() or ["TCS_NS"]
TICKER_TO_ID = {t: i for i, t in enumerate(TICKER_LIST)}


def _sector_for_ticker_name(ticker: str) -> str:
    """Lazy import so dataset stays usable without the features package in minimal tests."""
    from features.sectors import get_sector

    return get_sector(ticker)


def _regime_fallback_from_ret1d(ret1d: pd.Series) -> np.ndarray:
    """
    If HMM_Regime is missing, approximate a 3-way regime from trailing volatility of Ret_1d.
    Maps to {0,1,2} compatible with the model's regime embedding.
    """
    rv = ret1d.astype(float).rolling(20, min_periods=5).std()
    if rv.isna().any():
        raise ValueError(
            "HMM_Regime is missing and causal fallback would require non-causal backfill. "
            "Regenerate processed features with a valid HMM_Regime column."
        )
    try:
        # Terciles: low vol ~ bull(1), mid ~ high_vol(0), high vol ~ bear(2) — order by vol ascending
        q = pd.qcut(rv, q=3, labels=[1, 0, 2], duplicates="drop")
        out = pd.to_numeric(q, errors="coerce").fillna(1.0).astype(np.int64).values
        return out
    except Exception:
        med = float(rv.median())
        hi = rv > med * 1.25
        lo = rv < med * 0.75
        out = np.where(hi, 2, np.where(lo, 1, 0)).astype(np.int64)
        return out


def ensure_hmm_regime_array(df: pd.DataFrame) -> np.ndarray:
    """
    Return int64 regime per row. Prefer column HMM_Regime when present and valid;
    otherwise compute on-the-fly from Ret_1d.
    """
    n = len(df)
    if HMM_REGIME_COL in df.columns and df[HMM_REGIME_COL].notna().any():
        s = df[HMM_REGIME_COL].astype(float)
        if s.notna().all():
            return np.clip(s.round().astype(np.int64), 0, 2)
        filled = s.fillna(1.0)
        return np.clip(filled.round().astype(np.int64), 0, 2)

    if "Ret_1d" not in df.columns:
        return np.ones(n, dtype=np.int64)
    return _regime_fallback_from_ret1d(df["Ret_1d"])


def clip_target(y, lo: float = -0.2, hi: float = 0.2):
    """
    Clip extreme label values to reduce damage from bad data (splits, spikes).
    Monthly alpha outside ±20% is treated as suspicious for this project and capped.
    """
    import numpy as _np

    arr = _np.asarray(y, dtype=_np.float32)
    return _np.clip(arr, lo, hi)


class SequenceDataset(Dataset):
    """
    Creates overlapping sequences of length seq_len from
    a numpy feature array. Each sample is:
      X: (seq_len, num_features) — input sequence
      y: scalar — target
      regime: integer — HMM_Regime at last timestep
    """

    def __init__(self, features, targets, regimes, seq_len=30, ticker_id=None):
        self.features = features  # numpy (N, num_features)
        self.targets = targets  # numpy (N,)
        self.regimes = regimes  # numpy (N,) integers
        self.seq_len = seq_len
        self.ticker_id = ticker_id
        self.valid_idx = list(range(seq_len, len(targets)))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        X = self.features[i - self.seq_len : i]
        y = self.targets[i]
        regime = self.regimes[i]
        batch = (
            torch.FloatTensor(X),
            torch.FloatTensor([y]),
            torch.LongTensor([regime]),
        )
        if self.ticker_id is None:
            return batch
        return batch + (torch.LongTensor([int(self.ticker_id)]),)


class RAMTDataset:
    """
    Loads one ticker's processed Parquet features, applies RobustScaler fit on train only
    via get_fold_loaders. Primary target: Monthly_Alpha.

    Key guarantee: RobustScaler is fit on training data only — zero leakage.
    """

    def __init__(
        self,
        ticker,
        data_dir="data/processed",
        seq_len=30,
        batch_size=32,
    ):
        self.ticker = ticker
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.scaler = RobustScaler()
        self.df = None
        self.features = None
        self.targets = None
        self.regimes = None
        self.ticker_id = None
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        """Load Parquet; RobustScaler features = ALL_FEATURE_COLS only; regime separate."""
        path = os.path.join(data_dir, f"{self.ticker}_features.parquet")
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True).set_index("Date", drop=True)

        missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for {self.ticker}: {missing}")

        if TARGET_COL in df.columns and df[TARGET_COL].notna().any():
            eff = TARGET_COL
        elif TARGET_COL == "Sector_Alpha" and "Monthly_Alpha" in df.columns:
            warnings.warn(
                f"{self.ticker}: Sector_Alpha is missing/all-NaN; using Monthly_Alpha.",
                RuntimeWarning,
                stacklevel=2,
            )
            eff = "Monthly_Alpha"
        else:
            raise ValueError(f"Missing usable target {TARGET_COL} for {self.ticker}")

        df = df.dropna(subset=[eff])
        # Daily_Return optional for single-ticker loaders; multi-ticker training sets it
        if "Daily_Return" in df.columns:
            df = df.dropna(subset=["Daily_Return"])

        regime_arr = ensure_hmm_regime_array(df.reset_index())

        self.df = df
        self.dates = df.index.values
        self.features_raw = df[list(ALL_FEATURE_COLS)].values.astype(np.float32)
        self.targets = df[eff].values.astype(np.float32)
        self.regimes = regime_arr.astype(np.int64)

    def get_fold_loaders(self, train_idx, test_idx, val_fraction=0.15):
        val_size = max(1, int(len(train_idx) * val_fraction))
        actual_train_idx = train_idx[:-val_size]
        val_idx = train_idx[-val_size:]

        X_train = self.features_raw[actual_train_idx]
        X_val = self.features_raw[val_idx]
        X_test = self.features_raw[test_idx]

        y_train = self.targets[actual_train_idx]
        y_val = self.targets[val_idx]
        y_test = self.targets[test_idx]

        r_train = self.regimes[actual_train_idx]
        r_val = self.regimes[val_idx]
        r_test = self.regimes[test_idx]

        self.scaler = RobustScaler()
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)
        X_test_sc = self.scaler.transform(X_test)

        train_ds = SequenceDataset(
            X_train_sc, y_train, r_train, self.seq_len, ticker_id=self.ticker_id
        )
        val_ds = SequenceDataset(X_val_sc, y_val, r_val, self.seq_len, ticker_id=self.ticker_id)
        test_ds = SequenceDataset(
            X_test_sc, y_test, r_test, self.seq_len, ticker_id=self.ticker_id
        )

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


# Backward-compatible name used in model.py / encoder.py
RAMTDataModule = RAMTDataset


class LazyTickerStore:
    """
    LRU-cached ticker parquet reader. Scaled feature matrix uses ALL_FEATURE_COLS only.
    Regime is separate (HMM_Regime or fallback).
    """

    def __init__(self, processed_dir: str = "data/processed", cache_size: int = 6):
        self.processed_dir = Path(processed_dir)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, dict[str, object]] = OrderedDict()

    def _load_ticker_df(self, ticker: str) -> pd.DataFrame:
        path = self.processed_dir / f"{ticker}_features.parquet"
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def get(self, ticker: str) -> dict[str, object]:
        if ticker in self._cache:
            self._cache.move_to_end(ticker)
            return self._cache[ticker]

        df = self._load_ticker_df(ticker)
        missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker}: missing feature columns: {missing}")
        if TARGET_COL in df.columns and df[TARGET_COL].notna().any():
            eff_target = TARGET_COL
        elif TARGET_COL == "Sector_Alpha" and "Monthly_Alpha" in df.columns:
            warnings.warn(
                f"{ticker}: Sector_Alpha is missing/all-NaN; using Monthly_Alpha until you "
                "re-run features/feature_engineering.py (sector-neutral panel step).",
                RuntimeWarning,
                stacklevel=2,
            )
            eff_target = "Monthly_Alpha"
        else:
            raise ValueError(f"{ticker}: missing usable {TARGET_COL}")

        # Legacy / partial Parquet may omit Daily_Return; match features/feature_engineering.add_daily_target
        if "Daily_Return" not in df.columns:
            if "Ret_1d" not in df.columns:
                raise ValueError(f"{ticker}: need Daily_Return or Ret_1d to build tactical target")
            df = df.copy()
            df["Daily_Return"] = df["Ret_1d"].shift(-1)

        df = df.dropna(subset=[eff_target, "Daily_Return"])

        dates = pd.DatetimeIndex(df["Date"])
        X = df[list(ALL_FEATURE_COLS)].values.astype(np.float32)
        y_m = df[eff_target].values.astype(np.float32)
        y_d = df["Daily_Return"].values.astype(np.float32)
        r = ensure_hmm_regime_array(df)

        item = {"dates": dates, "X": X, "y_m": y_m, "y_d": y_d, "r": r}
        self._cache[ticker] = item
        self._cache.move_to_end(ticker)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item


class LazyMultiTickerSequenceDataset(Dataset):
    """
    Sequence dataset backed by per-ticker Parquet files, loaded on demand.
    sample_keys: list of (ticker, index) where index refers to row position within that ticker.

    When ``feature_scaler`` / ``y_scaler`` are provided (fitted on training data only),
    each sample applies ``.transform()`` to the raw Parquet window so train/val/test
    match inference (Systems-First scaling bridge).
    """

    def __init__(
        self,
        store: LazyTickerStore,
        sample_keys: list[tuple[str, int]],
        seq_len: int = 30,
        feature_scaler: RobustScaler | Any = None,
        y_scaler: RobustScaler | None = None,
        y_winsor_lo: float | None = None,
        y_winsor_hi: float | None = None,
    ):
        self.store = store
        self.sample_keys = sample_keys
        self.seq_len = int(seq_len)
        self.feature_scaler = feature_scaler  # RobustScaler or SectorNeutralScaler (set_active_sector)
        self.y_scaler = y_scaler
        self.y_winsor_lo = y_winsor_lo
        self.y_winsor_hi = y_winsor_hi

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        ticker, i = self.sample_keys[idx]
        td = self.store.get(ticker)
        X_raw = td["X"][i - self.seq_len : i]
        if self.feature_scaler is not None:
            fs = self.feature_scaler
            if hasattr(fs, "set_active_sector"):
                fs.set_active_sector(_sector_for_ticker_name(ticker))
            X = fs.transform(X_raw.astype(np.float64, copy=False)).astype(np.float32)
        else:
            X = np.asarray(X_raw, dtype=np.float32)

        y_m_raw = float(td["y_m"][i])
        if self.y_winsor_lo is not None and self.y_winsor_hi is not None:
            y_m_raw_w = float(np.clip(y_m_raw, self.y_winsor_lo, self.y_winsor_hi))
        else:
            y_m_raw_w = y_m_raw
        if self.y_scaler is not None:
            y_m = float(
                self.y_scaler.transform(np.array([[y_m_raw_w]], dtype=np.float64))[0, 0]
            )
        else:
            y_m = y_m_raw_w

        y_d = float(td["y_d"][i])
        r = int(td["r"][i])
        d = int(td["dates"][i].value)
        tid = int(TICKER_TO_ID.get(ticker, 0))
        # Winsorized raw monthly target (for ranking loss in unscaled alpha units)
        return (
            torch.from_numpy(X),
            torch.tensor([y_m], dtype=torch.float32),
            torch.tensor([y_d], dtype=torch.float32),
            torch.tensor([r], dtype=torch.long),
            torch.tensor([tid], dtype=torch.long),
            torch.tensor([d], dtype=torch.long),
            torch.tensor([y_m_raw_w], dtype=torch.float32),
        )


if __name__ == "__main__":
    print("Testing RAMTDataset...")

    dm = RAMTDataset("TCS_NS", seq_len=30, batch_size=32)
    folds = dm.get_walk_forward_indices()

    print(f"Total folds: {len(folds)}")
    print(f"Total rows: {len(dm.targets)}")
    print(f"Feature columns ({len(ALL_FEATURE_COLS)}): {ALL_FEATURE_COLS}")

    train_idx, test_idx = folds[0]
    train_loader, val_loader, test_loader, dates = dm.get_fold_loaders(train_idx, test_idx)

    X_batch, y_batch, r_batch = next(iter(train_loader))
    print("\nFirst fold:")
    print(f"  X shape: {X_batch.shape}")
    print(f"  y shape: {y_batch.shape}")
    print(f"  regime shape: {r_batch.shape}")
    print("\nAll checks passed.")
