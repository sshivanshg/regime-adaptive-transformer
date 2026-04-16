"""
RAMT Ranking Model Training

Trains on ALL NIFTY 50 stocks combined.
Target: Monthly alpha vs NIFTY (beat/miss benchmark)
Loss: MSE on monthly alpha + ranking loss
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ramt.dataset import (
    ALL_FEATURE_COLS,
    LazyMultiTickerSequenceDataset,
    LazyTickerStore,
    TARGET_COL,
    TICKER_TO_ID,
    build_ticker_universe,
)
from models.ramt.losses import CombinedLoss, TournamentRankingLoss
from models.ramt.model import build_ramt
def _winsorize_with_bounds(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Soft clipping (winsorization): cap values at precomputed bounds.
    Bounds must be computed from TRAIN ONLY to avoid leakage.
    """
    return np.clip(y.astype(np.float32), float(lo), float(hi))


def _fit_y_scaler_on_train(
    data: dict[str, TickerData], train_keys: list[tuple[str, int]]
) -> RobustScaler:
    """
    Fit a RobustScaler on the MONTHLY label (after clipping) for stability.
    """
    ys = []
    for t, i in train_keys:
        td = data[t]
        ys.append(td.y_monthly_raw[i])
    y_arr = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
    sc = RobustScaler()
    sc.fit(y_arr)
    return sc


def _apply_y_scaler(data: dict[str, TickerData], y_scaler: RobustScaler) -> dict[str, TickerData]:
    out: dict[str, TickerData] = {}
    for t, td in data.items():
        y_sc = y_scaler.transform(td.y_monthly_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)
        out[t] = TickerData(
            ticker=td.ticker,
            ticker_id=td.ticker_id,
            dates=td.dates,
            X=td.X,
            y_monthly=y_sc,
            y_daily=td.y_daily,
            y_monthly_raw=td.y_monthly_raw,
            y_daily_raw=td.y_daily_raw,
            regime=td.regime,
            sector=getattr(td, "sector", "OTHER"),
        )
    return out


TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2026-04-15"

TICKERS = build_ticker_universe("data/processed")


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

SEQ_LEN = 30
BATCH_SIZE = 64
# Post–warm-up LR; warmup ramps from WARMUP_LR_START → WARMUP_LR_END over WARMUP_STEPS optimizer steps.
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
WARMUP_LR_START = 1e-7
WARMUP_LR_END = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 8
GRAD_CLIP = 1.0
LAMBDA_DIR = 0.3
NUM_HEADS = 8
MODEL_DROPOUT = 0.2
HIGH_VOL_SAMPLE_WEIGHT = 2.0  # regime 0
# MarginRankingLoss margin: higher → model must separate leaders from the pack more aggressively.
RANKING_MARGIN = 3.0

# --- Pessimism-bias fix (plan: eager-rolling-sphinx) -----------------------
# See /Users/shivanshgupta/.claude/plans/eager-rolling-sphinx.md for the
# mathematical defect summary that motivated these knobs.
USE_TOURNAMENT_LOSS = True            # full-pairwise magnitude-weighted ranking
# Margin in unscaled monthly-alpha units (winsorized % space); paired with inverse-scaled preds.
RANKING_MARGIN_ALPHA = 0.02
AUX_DAILY_WEIGHT = 0.05               # tiny MSE anchor on daily head (was 0.3)
MIN_CROSSSECTION_SIZE = 8             # min stocks/date for ranking loss (was 4)

# --- Sector-neutral scaling (plan: Upgrade 3 / D3) -------------------------
# "sector" = per-sector RobustScaler with global fallback; "global" = legacy
SCALER_MODE = "sector"


def _safe_ticker_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    if stem.endswith("_features"):
        stem = stem[: -len("_features")]
    return stem


def _rebalance_dates_21d(
    nifty_raw_path: str, start: str, end: str, step_size: int = 21
) -> pd.DatetimeIndex:
    """
    Rebalance every `step_size` trading days using NIFTY trading calendar.
    """
    p = Path(nifty_raw_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p, columns=["Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
        dates = pd.DatetimeIndex(df["Date"].unique())
    else:
        df = pd.read_csv(nifty_raw_path, parse_dates=["Date"]).sort_values("Date")
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
        dates = pd.DatetimeIndex(df["Date"].unique())
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    return dates[::step_size]


def _nifty_raw_path() -> str:
    root = Path("data/raw")
    pq = root / "_NSEI.parquet"
    if pq.is_file():
        return str(pq)
    csv = root / "_NSEI_raw.csv"
    if csv.is_file():
        return str(csv)
    raise FileNotFoundError(f"Expected {pq} or {csv}")


def _full_nifty_trading_calendar(nifty_path: str) -> pd.DatetimeIndex:
    p = Path(nifty_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p, columns=["Date"])
    else:
        df = pd.read_csv(p, usecols=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.DatetimeIndex(df["Date"].sort_values().unique())


def _last_trading_day_before(cal: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    sub = cal[cal < pd.Timestamp(ts)]
    if len(sub) == 0:
        return pd.Timestamp(ts)
    return pd.Timestamp(sub[-1])


def _train_ramt_combined_fold(
    store: LazyTickerStore,
    tickers: list[str],
    train_start: str,
    train_end_inclusive: str,
    max_epochs: int | None,
    plot_dir: str | None,
    save_artifacts: bool,
    fold_label: str,
) -> tuple[object, Union[RobustScaler, "SectorNeutralScaler"], RobustScaler, float, float]:
    """
    Fit scaler + y transforms on training keys only, then train RAMT (train + val loaders).
    """
    train_keys = _build_sample_keys_from_store(store, tickers, train_start, train_end_inclusive, SEQ_LEN)
    val_start = (pd.Timestamp(train_end_inclusive) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    val_keys = _build_sample_keys_from_store(store, tickers, val_start, train_end_inclusive, SEQ_LEN)
    train_keys_final = [k for k in train_keys if k not in set(val_keys)]

    from features.sectors import get_sector

    data_sc: dict[str, TickerData] = {}
    for t in tickers:
        td = store.get(t)
        data_sc[t] = TickerData(
            ticker=t,
            ticker_id=int(TICKER_TO_ID.get(t, 0)),
            dates=td["dates"],  # type: ignore[arg-type]
            X=td["X"],  # type: ignore[arg-type]
            y_monthly=td["y_m"],  # type: ignore[arg-type]
            y_daily=td["y_d"],  # type: ignore[arg-type]
            y_monthly_raw=td["y_m"],  # type: ignore[arg-type]
            y_daily_raw=td["y_d"],  # type: ignore[arg-type]
            regime=td["r"],  # type: ignore[arg-type]
            sector=get_sector(t),
        )

    if len(train_keys_final) < 100:
        raise RuntimeError(
            f"{fold_label}: insufficient training keys ({len(train_keys_final)}). "
            f"train_end_inclusive={train_end_inclusive}"
        )

    if SCALER_MODE == "sector":
        scaler = _fit_sector_neutral_scaler_on_train(data_sc, train_keys_final)
    else:
        scaler = _fit_scaler_on_train(data_sc, train_keys_final)

    y_train_raw = np.asarray(
        [float(data_sc[t].y_monthly_raw[i]) for t, i in train_keys_final], dtype=np.float32
    )
    lo_b = float(np.nanpercentile(y_train_raw, 1.0))
    hi_b = float(np.nanpercentile(y_train_raw, 99.0))
    patched_w: dict[str, TickerData] = {}
    for t, td in data_sc.items():
        y_raw_w = _winsorize_with_bounds(td.y_monthly_raw, lo_b, hi_b)
        patched_w[t] = TickerData(
            ticker=td.ticker,
            ticker_id=td.ticker_id,
            dates=td.dates,
            X=td.X,
            y_monthly=y_raw_w.copy(),
            y_daily=td.y_daily,
            y_monthly_raw=y_raw_w,
            y_daily_raw=td.y_daily_raw,
            regime=td.regime,
            sector=td.sector,
        )
    data_sc = patched_w

    y_scaler = _fit_y_scaler_on_train(data_sc, train_keys_final)

    train_ds = MultiTickerSequenceDataset(
        store,
        sorted(train_keys_final),
        SEQ_LEN,
        feature_scaler=scaler,
        y_scaler=y_scaler,
        y_winsor_lo=lo_b,
        y_winsor_hi=hi_b,
    )
    val_ds = MultiTickerSequenceDataset(
        store,
        sorted(val_keys),
        SEQ_LEN,
        feature_scaler=scaler,
        y_scaler=y_scaler,
        y_winsor_lo=lo_b,
        y_winsor_hi=hi_b,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    model = build_ramt({"seq_len": SEQ_LEN, "num_heads": NUM_HEADS, "dropout": MODEL_DROPOUT}).to(DEVICE)
    criterion = CombinedLoss(lambda_dir=LAMBDA_DIR)
    optimizer = AdamW(model.parameters(), lr=WARMUP_LR_END, weight_decay=WEIGHT_DECAY)
    plateau = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )
    global_step = [0]

    n_epochs = int(max_epochs) if max_epochs is not None else int(MAX_EPOCHS)
    epoch_nums: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    lr_snapshots: list[float] = []

    print(
        f"{fold_label}: train_samples={len(train_ds)} val_samples={len(val_ds)} epochs<={n_epochs}",
        flush=True,
    )

    best = float("inf")
    best_state = None
    patience_ctr = 0
    for epoch in range(n_epochs):
        tr_m = _train_one_epoch(
            model, train_loader, optimizer, criterion, global_step=global_step, y_scaler=y_scaler
        )
        v = _eval_loss(model, val_loader, criterion, y_scaler=y_scaler)
        if global_step[0] >= WARMUP_STEPS:
            plateau.step(v)
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_nums.append(epoch + 1)
        train_losses.append(float(tr_m))
        val_losses.append(float(v))
        lr_snapshots.append(lr_now)
        print(
            f"  [{fold_label}] epoch {epoch + 1:02d}/{n_epochs}  train_loss={tr_m:.6f}  "
            f"val_loss={v:.6f}  lr={lr_now:.2e}",
            flush=True,
        )
        if v < best:
            best = v
            patience_ctr = 0
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    if plot_dir and save_artifacts:
        _save_training_run_artifacts(
            Path(plot_dir),
            epoch_nums,
            train_losses,
            val_losses,
            lr_snapshots,
            int(global_step[0]),
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, y_scaler, lo_b, hi_b


def _predict_rows_for_dates(
    dates: pd.DatetimeIndex,
    *,
    store: LazyTickerStore,
    tickers: list[str],
    model: object,
    scaler: Union[RobustScaler, "SectorNeutralScaler"],
    y_scaler: RobustScaler,
    lo_b: float,
    hi_b: float,
    inference_warmup_days: int,
) -> list[dict[str, object]]:
    """Inference on fixed calendar dates; requires SEQ_LEN + warmup rows of history per ticker."""
    min_i = SEQ_LEN + int(inference_warmup_days)
    rows: list[dict[str, object]] = []
    for d in dates:
        dts = pd.Timestamp(d)
        period = "Train" if dts <= pd.Timestamp(TRAIN_END) else "Test"
        for t in tickers:
            td = store.get(t)
            d_arr = td["dates"]
            try:
                i = int(d_arr.get_loc(dts))
            except KeyError:
                continue
            if i < min_i:
                continue
            X_raw = td["X"][i - SEQ_LEN : i]
            if hasattr(scaler, "set_active_sector"):
                from features.sectors import get_sector

                scaler.set_active_sector(get_sector(t))
            Xseq = (
                torch.from_numpy(scaler.transform(X_raw.astype(np.float64)).astype(np.float32))
                .float()
                .unsqueeze(0)
                .to(DEVICE)
            )
            r = torch.tensor([int(td["r"][i])], dtype=torch.long).to(DEVICE)
            tid = torch.tensor([int(TICKER_TO_ID.get(t, 0))], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                pred_m_sc, _pred_d, _g = model(Xseq, r, ticker_id=tid)
            pred_m = float(y_scaler.inverse_transform([[float(pred_m_sc.cpu().numpy().squeeze())]])[0][0])
            y_raw = float(td["y_m"][i])
            actual_w = float(np.clip(y_raw, lo_b, hi_b))
            rows.append(
                {
                    "Date": dts,
                    "Ticker": t,
                    "predicted_alpha": float(pred_m),
                    "actual_alpha": actual_w,
                    "Period": period,
                }
            )
    return rows


@dataclass(frozen=True)
class TickerData:
    ticker: str
    ticker_id: int
    dates: pd.DatetimeIndex
    X: np.ndarray  # (N, F) float32
    y_monthly: np.ndarray  # (N,) float32
    y_daily: np.ndarray  # (N,) float32
    y_monthly_raw: np.ndarray  # (N,) float32 (unscaled, clipped)
    y_daily_raw: np.ndarray  # (N,) float32 (unscaled)
    regime: np.ndarray  # (N,) int64
    sector: str = "OTHER"


class MultiTickerSequenceDataset(LazyMultiTickerSequenceDataset):
    """Backward-compatible alias; now lazy-loaded from parquet."""
    pass


def _build_sample_keys(
    td: TickerData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seq_len: int,
) -> list[tuple[str, int]]:
    # indices where date in [start, end)
    mask = (td.dates >= start) & (td.dates < end)
    idxs = np.where(mask)[0]
    idxs = idxs[idxs >= seq_len]
    return [(td.ticker, int(i)) for i in idxs]


def _build_sample_keys_from_store(
    store: LazyTickerStore, tickers: list[str], start: str, end: str, seq_len: int
) -> list[tuple[str, int]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    keys: list[tuple[str, int]] = []
    for t in tickers:
        td = store.get(t)
        dates: pd.DatetimeIndex = td["dates"]  # type: ignore[assignment]
        mask = (dates >= start_ts) & (dates < end_ts)
        idxs = np.where(mask)[0]
        idxs = idxs[idxs >= seq_len]
        keys.extend([(t, int(i)) for i in idxs])
    return keys


def _fit_scaler_on_train(
    data: dict[str, TickerData],
    train_keys: list[tuple[str, int]],
    max_fit_samples: int = 200_000,
) -> RobustScaler:
    """
    Fit a RobustScaler on training features.

    RobustScaler does not support partial_fit, so we fit on a deterministic
    subset of training samples to keep memory bounded.
    """
    scaler = RobustScaler()
    if len(train_keys) == 0:
        scaler.fit(np.empty((0, len(ALL_FEATURE_COLS)), dtype=np.float32))
        return scaler

    step = max(1, int(len(train_keys) / max_fit_samples))
    chosen = train_keys[::step]
    Xb = np.asarray([data[t].X[i] for t, i in chosen], dtype=np.float32)
    scaler.fit(Xb)
    return scaler


class SectorNeutralScaler:
    """
    Per-sector RobustScaler with a global fallback for small/unknown sectors.

    Rationale: a 1σ RSI swing in BANK should be comparable to a 1σ swing in IT.
    Pooling them (as the legacy global scaler does) compresses high-vol sectors
    and amplifies low-vol ones — the sector-alpha-washout defect (D3 in plan).

    The fit requires a companion `sectors` array aligned with `X` rows. Lookup
    of sector-for-ticker is resolved via features.sectors.get_sector.

    Intended to be ``sklearn``-API compatible enough for drop-in use in
    LazyMultiTickerSequenceDataset — exposes ``.transform(X)`` that routes
    by the currently-loaded ticker's sector (set via ``set_active_sector``).
    """

    def __init__(self, min_samples_per_sector: int = 500):
        self.min_samples_per_sector = int(min_samples_per_sector)
        self.per_sector: dict[str, RobustScaler] = {}
        self.global_fallback: RobustScaler = RobustScaler()
        self._active_sector: str | None = None

    # --- fit / route --------------------------------------------------------
    def fit(self, X: np.ndarray, sectors: np.ndarray) -> "SectorNeutralScaler":
        X = np.asarray(X, dtype=np.float32)
        sectors = np.asarray(sectors)
        if X.shape[0] == 0:
            self.global_fallback.fit(np.empty((0, X.shape[1] if X.ndim == 2 else 1),
                                              dtype=np.float32))
            return self
        self.global_fallback.fit(X)
        for sec in np.unique(sectors):
            mask = sectors == sec
            if int(mask.sum()) >= self.min_samples_per_sector:
                sc = RobustScaler()
                sc.fit(X[mask])
                self.per_sector[str(sec)] = sc
        return self

    def set_active_sector(self, sector: str | None) -> None:
        """Route ``.transform`` calls to a specific sector's scaler."""
        self._active_sector = sector

    # --- sklearn-style transform -------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X using the currently-active sector scaler. Falls back to
        the global scaler if the sector is unknown or under-sampled.
        """
        sc = self.per_sector.get(str(self._active_sector), self.global_fallback) \
            if self._active_sector is not None else self.global_fallback
        return sc.transform(X)

    def transform_for_sector(self, X: np.ndarray, sector: str) -> np.ndarray:
        sc = self.per_sector.get(str(sector), self.global_fallback)
        return sc.transform(X)


def _fit_sector_neutral_scaler_on_train(
    data: dict[str, TickerData],
    train_keys: list[tuple[str, int]],
    max_fit_samples: int = 200_000,
) -> SectorNeutralScaler:
    """
    Sector-neutral variant of _fit_scaler_on_train.

    Requires ``TickerData`` to carry a ``.sector`` attribute; ``get_sector``
    resolves it from the ticker stem when not populated.
    """
    from features.sectors import get_sector

    scaler = SectorNeutralScaler()
    if len(train_keys) == 0:
        scaler.fit(np.empty((0, len(ALL_FEATURE_COLS)), dtype=np.float32),
                   np.empty((0,), dtype=object))
        return scaler

    step = max(1, int(len(train_keys) / max_fit_samples))
    chosen = train_keys[::step]
    Xb_list: list[np.ndarray] = []
    sec_list: list[str] = []
    for t, i in chosen:
        td = data[t]
        Xb_list.append(td.X[i])
        sec_list.append(getattr(td, "sector", None) or get_sector(t))
    Xb = np.asarray(Xb_list, dtype=np.float32)
    sec_arr = np.asarray(sec_list, dtype=object)
    scaler.fit(Xb, sec_arr)
    return scaler


def _apply_scaler(
    data: dict[str, TickerData], scaler: Union[RobustScaler, "SectorNeutralScaler"]
) -> dict[str, TickerData]:
    out: dict[str, TickerData] = {}
    for t, td in data.items():
        if hasattr(scaler, "set_active_sector"):
            scaler.set_active_sector(str(td.sector))
        Xs = scaler.transform(td.X).astype(np.float32)
        out[t] = TickerData(
            ticker=td.ticker,
            ticker_id=td.ticker_id,
            dates=td.dates,
            X=Xs,
            y_monthly=td.y_monthly,
            y_daily=td.y_daily,
            y_monthly_raw=td.y_monthly_raw,
            y_daily_raw=td.y_daily_raw,
            regime=td.regime,
            sector=td.sector,
        )
    return out


def _pairwise_rank_loss(pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Pairwise ranking loss (logistic) for a set of items from the same rebalance date.
    Encourages correct ordering when y_true differs.
    """
    # pred, y_true: (n, 1)
    y = y_true.squeeze(-1)
    p = pred.squeeze(-1)
    # choose pairs by sorting and taking extremes to keep it cheap
    n = y.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=pred.device)
    k = min(10, n // 2)
    top_idx = torch.topk(y, k=k, largest=True).indices
    bot_idx = torch.topk(y, k=k, largest=False).indices
    top_p = p[top_idx].unsqueeze(1)
    bot_p = p[bot_idx].unsqueeze(0)
    # want top_p > bot_p
    margin = top_p - bot_p
    return torch.nn.functional.softplus(-margin).mean()


def _margin_rank_loss(pred: torch.Tensor, y_true: torch.Tensor, margin: float = RANKING_MARGIN) -> torch.Tensor:
    """
    MarginRankingLoss for a rebalance-date group.

    We only care that "A > B", not the exact alpha gap.
    This often avoids flat regression-like predictions on noisy targets.
    """
    y = y_true.squeeze(-1)
    p = pred.squeeze(-1)
    n = int(y.shape[0])
    if n < 4:
        return torch.tensor(0.0, device=pred.device)

    k = min(10, n // 2)
    top_idx = torch.topk(y, k=k, largest=True).indices
    bot_idx = torch.topk(y, k=k, largest=False).indices
    top_p = p[top_idx].unsqueeze(1)  # (k,1)
    bot_p = p[bot_idx].unsqueeze(0)  # (1,k)

    # Expand into all pairs: want top_p > bot_p
    x1 = top_p.expand(-1, bot_p.shape[1]).reshape(-1)
    x2 = bot_p.expand(top_p.shape[0], -1).reshape(-1)
    target = torch.ones_like(x1)
    crit = torch.nn.MarginRankingLoss(margin=margin)
    return crit(x1, x2, target)


def _time_decay_weights(db: torch.Tensor) -> torch.Tensor:
    """
    Time-decay weights:
    - 2024–2026: 2.0x
    - 2020: 0.5x
    - otherwise: 1.0x
    """
    years = pd.to_datetime(db.detach().cpu().numpy()).year.astype(np.int32)
    w = np.ones_like(years, dtype=np.float32)
    w = np.where(years == 2020, 0.5, w)
    w = np.where((years >= 2024) & (years <= 2026), 2.0, w)
    return torch.from_numpy(w).to(db.device)


def _dcg_gain(rel: torch.Tensor) -> torch.Tensor:
    # Standard gain used in NDCG: 2^rel - 1. Here rel is continuous; this is still usable.
    return torch.pow(2.0, rel) - 1.0


def _lambdarank_loss(pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    LambdaRank-style loss for a single rebalance-date group.

    Goal: get the TOP of the list right (closer to NDCG optimization).
    We weight pairwise logistic loss by the absolute change in NDCG from swapping items.

    Notes:
    - y_true is continuous (Monthly_Alpha). We use it as a relevance signal.
    - This is an approximation but works well in practice for financial ranking.
    """
    y = y_true.squeeze(-1)
    p = pred.squeeze(-1)
    n = int(y.shape[0])
    if n < 4:
        return torch.tensor(0.0, device=pred.device)

    # Normalize relevance inside the group to stabilize gain magnitudes.
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    rel = y_norm

    # Predicted order positions
    order_pred = torch.argsort(p, descending=True)
    ranks_pred = torch.empty_like(order_pred, dtype=torch.long)
    ranks_pred[order_pred] = torch.arange(n, device=pred.device, dtype=torch.long)  # 0 is best

    # Ideal DCG for normalization (using true rel)
    order_ideal = torch.argsort(rel, descending=True)
    discounts_ideal = 1.0 / torch.log2(torch.arange(n, device=pred.device, dtype=torch.float32) + 2.0)
    idcg = (_dcg_gain(rel[order_ideal]) * discounts_ideal).sum().clamp_min(1e-8)

    # Compute pairwise delta NDCG weights
    # Weight_{ij} = |ΔNDCG| when swapping i and j under current predicted ranks.
    ri = ranks_pred.unsqueeze(1).float()
    rj = ranks_pred.unsqueeze(0).float()
    di = 1.0 / torch.log2(ri + 2.0)
    dj = 1.0 / torch.log2(rj + 2.0)

    gi = _dcg_gain(rel).unsqueeze(1)
    gj = _dcg_gain(rel).unsqueeze(0)
    delta_dcg = (gi - gj) * (di - dj)
    w = (delta_dcg.abs() / idcg).detach()

    # Only consider pairs where true rel differs
    s = torch.sign(rel.unsqueeze(1) - rel.unsqueeze(0))
    mask = s != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Pairwise logistic loss: log(1 + exp(-(p_i - p_j) * s_ij))
    pij = p.unsqueeze(1) - p.unsqueeze(0)
    loss_mat = torch.nn.functional.softplus(-(pij * s))

    # Apply weights and mask
    loss = (loss_mat * w)[mask].mean()
    return loss


_TOURNAMENT = TournamentRankingLoss(margin=RANKING_MARGIN_ALPHA)


def _monthly_pred_unscaled(pred_m: torch.Tensor, y_scaler: RobustScaler) -> torch.Tensor:
    """Map RobustScaler monthly target space back to winsorized % alpha units (differentiable)."""
    s = torch.as_tensor(
        np.asarray(y_scaler.scale_, dtype=np.float32).ravel()[0],
        device=pred_m.device,
        dtype=pred_m.dtype,
    )
    c = torch.as_tensor(
        np.asarray(y_scaler.center_, dtype=np.float32).ravel()[0],
        device=pred_m.device,
        dtype=pred_m.dtype,
    )
    return pred_m.squeeze(-1) * s + c


def _rank_term(
    pred_m: torch.Tensor,
    yb_m_scaled: torch.Tensor,
    yb_m_unscaled: torch.Tensor,
    y_scaler: RobustScaler,
) -> torch.Tensor:
    """
    One rebalance-date ranking loss: tournament on unscaled alpha vs legacy margin on scaled targets.
    """
    if USE_TOURNAMENT_LOSS:
        pu = _monthly_pred_unscaled(pred_m, y_scaler)
        return _TOURNAMENT(pu.unsqueeze(-1), yb_m_unscaled.unsqueeze(-1))
    return _margin_rank_loss(pred_m, yb_m_scaled, margin=RANKING_MARGIN)


def _log_pred_dist(
    pred: torch.Tensor,
    y: torch.Tensor,
    tag: str,
) -> None:
    """
    Pessimism-bias diagnostic. Prints μ/σ of predictions vs targets and their
    correlation. Fires the alarm when predicted mean collapses toward 0 while
    actual mean is meaningfully positive (the 2024-2026 rally scenario).
    """
    with torch.no_grad():
        p = pred.detach().cpu().numpy().ravel()
        t = y.detach().cpu().numpy().ravel()
        if p.size == 0 or t.size == 0:
            return
        cov = float(np.corrcoef(p, t)[0, 1]) if p.size >= 2 and p.std() > 0 and t.std() > 0 else 0.0
        alarm = ""
        if abs(t.mean()) > 1e-4 and abs(p.mean()) < 0.3 * abs(t.mean()):
            alarm = "  ⚠ PESSIMISM-ALARM"
        print(
            f"[{tag}] pred μ={p.mean():+.4f} σ={p.std():.4f}  "
            f"actual μ={t.mean():+.4f} σ={t.std():.4f}  "
            f"corr={cov:+.3f}{alarm}",
            flush=True,
        )


def _train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    lambda_rank: float = 0.2,
    global_step: list[int] | None = None,
    y_scaler: RobustScaler | None = None,
):
    model.train()
    total = 0.0
    n = 0
    pred_accum: list[torch.Tensor] = []
    actual_accum: list[torch.Tensor] = []
    pbar = tqdm(loader, desc="train", leave=False, mininterval=0.5)
    for Xb, yb_m, yb_d, rb, tb, db, yb_m_raw in pbar:
        if global_step is not None and global_step[0] < WARMUP_STEPS:
            lr = WARMUP_LR_START + (WARMUP_LR_END - WARMUP_LR_START) * (
                global_step[0] + 1
            ) / float(WARMUP_STEPS)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        Xb = Xb.to(DEVICE)
        yb_m = yb_m.to(DEVICE)
        yb_m_raw = yb_m_raw.to(DEVICE).squeeze(-1)
        yb_d = yb_d.to(DEVICE)
        rb = rb.squeeze(-1).to(DEVICE)
        tb = tb.squeeze(-1).to(DEVICE)
        db = db.squeeze(-1).to(DEVICE)

        optimizer.zero_grad()
        pred_m, pred_d, _ = model(Xb, rb, ticker_id=tb)

        time_w = _time_decay_weights(db).to(dtype=torch.float32)
        rank_losses: list[torch.Tensor] = []
        rank_w: list[torch.Tensor] = []
        assert y_scaler is not None
        for d in torch.unique(db):
            m = db == d
            if int(m.sum()) >= MIN_CROSSSECTION_SIZE:
                gw = time_w[m].mean().clamp_min(1e-8)
                rank_losses.append(
                    _rank_term(pred_m[m], yb_m[m], yb_m_raw[m], y_scaler) * gw
                )
                rank_w.append(gw)
        if rank_losses:
            rank_loss = torch.stack(rank_losses).sum() / (torch.stack(rank_w).sum() + 1e-8)
        else:
            rank_loss = torch.tensor(0.0, device=DEVICE)

        w = torch.ones_like(rb, dtype=torch.float32, device=DEVICE)
        w = torch.where(rb == 0, torch.tensor(HIGH_VOL_SAMPLE_WEIGHT, device=DEVICE), w)
        w = w * time_w
        mse_d = (((pred_d.squeeze(-1) - yb_d.squeeze(-1)) ** 2) * w).sum() / (w.sum() + 1e-8)

        if USE_TOURNAMENT_LOSS:
            loss = rank_loss + AUX_DAILY_WEIGHT * mse_d
        else:
            strategic = lambda_rank * rank_loss * (w.mean())
            loss = 0.7 * strategic + 0.3 * mse_d

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        if global_step is not None:
            global_step[0] += 1

        total += float(loss.item())
        n += 1
        pbar.set_postfix(
            loss=f"{float(loss.item()):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            refresh=False,
        )

        pred_accum.append(_monthly_pred_unscaled(pred_m, y_scaler).detach())
        actual_accum.append(yb_m_raw.detach())

    if pred_accum:
        _log_pred_dist(torch.cat(pred_accum), torch.cat(actual_accum), tag="train")

    return total / max(n, 1)


def _eval_loss(
    model,
    loader,
    criterion,
    y_scaler: RobustScaler,
    lambda_rank: float = 0.2,
):
    model.eval()
    total = 0.0
    n = 0
    pred_accum: list[torch.Tensor] = []
    actual_accum: list[torch.Tensor] = []
    with torch.no_grad():
        for Xb, yb_m, yb_d, rb, tb, db, yb_m_raw in loader:
            Xb = Xb.to(DEVICE)
            yb_m = yb_m.to(DEVICE)
            yb_m_raw = yb_m_raw.to(DEVICE).squeeze(-1)
            yb_d = yb_d.to(DEVICE)
            rb = rb.squeeze(-1).to(DEVICE)
            tb = tb.squeeze(-1).to(DEVICE)
            db = db.squeeze(-1).to(DEVICE)
            pred_m, pred_d, _ = model(Xb, rb, ticker_id=tb)

            time_w = _time_decay_weights(db).to(dtype=torch.float32)
            rank_losses: list[torch.Tensor] = []
            rank_w: list[torch.Tensor] = []
            for d in torch.unique(db):
                m = db == d
                if int(m.sum()) >= MIN_CROSSSECTION_SIZE:
                    gw = time_w[m].mean().clamp_min(1e-8)
                    rank_losses.append(
                        _rank_term(pred_m[m], yb_m[m], yb_m_raw[m], y_scaler) * gw
                    )
                    rank_w.append(gw)
            if rank_losses:
                rank_loss = torch.stack(rank_losses).sum() / (torch.stack(rank_w).sum() + 1e-8)
            else:
                rank_loss = torch.tensor(0.0, device=DEVICE)

            w = torch.ones_like(rb, dtype=torch.float32, device=DEVICE)
            w = torch.where(rb == 0, torch.tensor(HIGH_VOL_SAMPLE_WEIGHT, device=DEVICE), w)
            w = w * time_w
            mse_d = (((pred_d.squeeze(-1) - yb_d.squeeze(-1)) ** 2) * w).sum() / (w.sum() + 1e-8)

            if USE_TOURNAMENT_LOSS:
                batch_loss = rank_loss + AUX_DAILY_WEIGHT * mse_d
            else:
                strategic = lambda_rank * rank_loss * (w.mean())
                batch_loss = 0.7 * strategic + 0.3 * mse_d

            total += float(batch_loss.item())
            n += 1

            pred_accum.append(_monthly_pred_unscaled(pred_m, y_scaler))
            actual_accum.append(yb_m_raw)

            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    if pred_accum:
        _log_pred_dist(torch.cat(pred_accum), torch.cat(actual_accum), tag="val")

    return total / max(n, 1)


def _predict(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    actuals = []
    ticker_ids = []
    with torch.no_grad():
        for Xb, yb_m, _yb_d, rb, tb, _db, _yb_raw in loader:
            Xb = Xb.to(DEVICE)
            rb = rb.squeeze(-1).to(DEVICE)
            tb = tb.squeeze(-1).to(DEVICE)
            pred_m, _pred_d, _g = model(Xb, rb, ticker_id=tb)
            preds.append(pred_m.cpu().numpy().squeeze())
            actuals.append(yb_m.numpy().squeeze())
            ticker_ids.append(tb.cpu().numpy().squeeze())
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(ticker_ids)


def _save_training_run_artifacts(
    out_dir: Path,
    epoch_nums: list[int],
    train_losses: list[float],
    val_losses: list[float],
    lr_snapshots: list[float],
    global_step_end: int,
) -> None:
    """
    Persist CSV + a multi-panel matplotlib dashboard (loss + LR) for the blind-split run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "epoch": epoch_nums,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "lr": lr_snapshots,
        }
    )
    df.to_csv(out_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("RAMT blind-split training (combined NIFTY200)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(epoch_nums, train_losses, "o-", color="#2563eb", label="train", linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss (combined objective)")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epoch_nums, val_losses, "o-", color="#dc2626", label="validation", linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss")
    ax.set_title("Validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epoch_nums, lr_snapshots, "s-", color="#059669", linewidth=1.5, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR (warm-up + ReduceLROnPlateau)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    gap = np.array(val_losses, dtype=np.float64) - np.array(train_losses, dtype=np.float64)
    ax.bar(epoch_nums, gap, color="#7c3aed", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val − train")
    ax.set_title("Generalization gap (val loss − train loss)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.text(0.5, 0.02, f"optimizer steps (end): {global_step_end}", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    png_path = out_dir / "training_dashboard.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training plots: {png_path}  &  {out_dir / 'training_history.csv'}", flush=True)


def _artifact_tag(raw: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "fold"


def save_ramt_inference_artifacts(
    out_dir: Path,
    *,
    model: torch.nn.Module,
    scaler: Union[RobustScaler, "SectorNeutralScaler"],
    y_scaler: RobustScaler,
    train_start: str,
    train_end: str,
    lo_b: float,
    hi_b: float,
    fold_label: str,
    fold_tag: str | None = None,
) -> None:
    """
    Persist the inference artifacts expected by the dashboard and audit tools.

    The canonical filenames are overwritten each fold so live inference always uses the
    freshest expanding-window model. Fold-tagged snapshots are also written for debugging.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "seq_len": SEQ_LEN,
            "num_heads": NUM_HEADS,
            "dropout": MODEL_DROPOUT,
        },
        "train_start": str(train_start),
        "train_end": str(train_end),
        "fold_label": fold_label,
        "y_scaler_center": float(np.asarray(y_scaler.center_, dtype=float).ravel()[0]),
        "y_scaler_scale": float(np.asarray(y_scaler.scale_, dtype=float).ravel()[0]),
        "y_target_col": TARGET_COL,
        "y_winsor_lo": float(lo_b),
        "y_winsor_hi": float(hi_b),
    }

    state_path = out_dir / "ramt_model_state.pt"
    scaler_path = out_dir / "ramt_scaler.joblib"
    y_scaler_path = out_dir / "ramt_y_scaler.joblib"
    torch.save(payload, state_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    if fold_tag:
        tag = _artifact_tag(fold_tag)
        torch.save(payload, out_dir / f"ramt_model_state_{tag}.pt")
        joblib.dump(scaler, out_dir / f"ramt_scaler_{tag}.joblib")
        joblib.dump(y_scaler, out_dir / f"ramt_y_scaler_{tag}.joblib")

    print(
        f"Saved RAMT fold artifacts for {fold_label}: "
        f"{state_path.name}, {scaler_path.name}, {y_scaler_path.name}",
        flush=True,
    )


def combined_walk_forward(
    start: str = "2016-01-01",
    end: str = "2024-12-31",
    test_steps: int = 3,
    training_step: int = 126,
    rebalance_step: int = 21,
    step_size: int | None = None,
    inference_warmup_days: int = 30,
    max_epochs: int | None = None,
    plot_dir: str | None = "results",
    artifact_dir: str | None = None,
) -> pd.DataFrame:
    """
    Combined RAMT with **decoupled** walk-forward training vs inference cadence.

    - **training_step** (default 126 trading days ≈ 6 months): retrain the model each time
      the out-of-sample calendar advances by this step. Training uses all history strictly
      before the first prediction date of that segment (expanding window; after the first
      segment, labels from realized test months are included — standard production-style WF).
    - **rebalance_step** (default 21): inside each segment, inference runs on every
      rebalance date on this NIFTY trading-day grid (no training between these dates).
    - **inference_warmup_days** (default 30): require at least ``SEQ_LEN + warmup`` rows
      of history before scoring a ticker (stricter than sequence length alone).

    Legacy: if ``step_size`` is passed, it overrides ``rebalance_step`` for backward compatibility.
    """
    if step_size is not None:
        rebalance_step = int(step_size)

    store = LazyTickerStore("data/processed", cache_size=6)
    tickers = list(TICKERS)
    if not tickers:
        raise FileNotFoundError("No processed parquet feature files found under data/processed.")

    nifty_path = _nifty_raw_path()
    full_cal = _full_nifty_trading_calendar(nifty_path)

    # Outer calendar: 6-month (126d) test segments; inner: 21d predictions per segment.
    segment_starts = _rebalance_dates_21d(
        nifty_path, TEST_START, TEST_END, step_size=int(training_step)
    )
    if len(segment_starts) == 0:
        raise RuntimeError(
            f"No walk-forward segments for TEST_START={TEST_START} TEST_END={TEST_END} "
            f"training_step={training_step}"
        )

    all_rows: list[dict[str, object]] = []

    for seg_idx, seg_start in enumerate(segment_starts):
        if seg_idx == 0:
            train_end_inclusive = TRAIN_END
        else:
            train_end_inclusive = _last_trading_day_before(full_cal, pd.Timestamp(seg_start)).strftime(
                "%Y-%m-%d"
            )

        fold_label = f"WF seg {seg_idx + 1}/{len(segment_starts)} train_end<={train_end_inclusive}"

        model, scaler, y_scaler, lo_b, hi_b = _train_ramt_combined_fold(
            store,
            tickers,
            TRAIN_START,
            train_end_inclusive,
            max_epochs,
            plot_dir,
            save_artifacts=bool(plot_dir) and seg_idx == 0,
            fold_label=fold_label,
        )
        if artifact_dir:
            save_ramt_inference_artifacts(
                Path(artifact_dir),
                model=model,
                scaler=scaler,
                y_scaler=y_scaler,
                train_start=TRAIN_START,
                train_end=train_end_inclusive,
                lo_b=lo_b,
                hi_b=hi_b,
                fold_label=fold_label,
                fold_tag=f"wf_seg_{seg_idx + 1:02d}",
            )

        # Next segment start (exclusive upper bound for prediction dates in this fold)
        seg_end_ts = (
            pd.Timestamp(segment_starts[seg_idx + 1])
            if seg_idx + 1 < len(segment_starts)
            else pd.Timestamp(TEST_END) + pd.Timedelta(days=1)
        )
        pred_end = _last_trading_day_before(full_cal, seg_end_ts)
        pred_dates = _rebalance_dates_21d(
            nifty_path,
            str(pd.Timestamp(seg_start).date()),
            str(pred_end.date()),
            step_size=int(rebalance_step),
        )

        fold_rows = _predict_rows_for_dates(
            pred_dates,
            store=store,
            tickers=tickers,
            model=model,
            scaler=scaler,
            y_scaler=y_scaler,
            lo_b=lo_b,
            hi_b=hi_b,
            inference_warmup_days=inference_warmup_days,
        )
        all_rows.extend(fold_rows)

        # First segment only: also emit train-window predictions (same model, 21d grid).
        if seg_idx == 0:
            train_pred_dates = _rebalance_dates_21d(
                nifty_path, TRAIN_START, TRAIN_END, step_size=int(rebalance_step)
            )
            train_rows = _predict_rows_for_dates(
                train_pred_dates,
                store=store,
                tickers=tickers,
                model=model,
                scaler=scaler,
                y_scaler=y_scaler,
                lo_b=lo_b,
                hi_b=hi_b,
                inference_warmup_days=inference_warmup_days,
            )
            all_rows.extend(train_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return df.sort_values(["Date", "predicted_alpha"], ascending=[True, False])


def train_fixed_and_predict(
    train_start: str = "2016-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2025-12-31",
    step_size: int = 21,
    max_epochs: int | None = None,
    target_col: str = "Monthly_Alpha",
) -> pd.DataFrame:
    """
    Train ONE combined RAMT model on [train_start, train_end], then predict
    on rebalance dates in [test_start, test_end].

    This enforces a strict no-lookahead split for final backtests.
    """
    raise NotImplementedError(
        "train_fixed_and_predict() is deprecated in the Parquet-only NIFTY200 pipeline. "
        "Use combined_walk_forward() which now implements the hard blind split (2015–2022 train, 2023–2026 test), "
        "RobustScaler fit on train only, and Period-labeled outputs."
    )
    # Reload y from chosen target column (keeps feature pipeline untouched)
    if target_col not in {"Monthly_Alpha", "Monthly_Alpha_Z"}:
        raise ValueError(f"Unsupported target_col={target_col}. Use Monthly_Alpha or Monthly_Alpha_Z.")
    # Patch y arrays in-place from disk for chosen target (cheap and explicit).
    patched: dict[str, TickerData] = {}
    for t, td in data.items():
        p = Path("data/processed") / f"{t}_features.csv"
        df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").set_index("Date", drop=True)
        if target_col not in df.columns:
            raise ValueError(f"{t}: missing {target_col}. Re-run features/feature_engineering.py")
        if "Daily_Return" not in df.columns:
            raise ValueError(f"{t}: missing Daily_Return. Re-run features/feature_engineering.py")
        df = df.dropna(subset=[target_col, "Daily_Return"])
        # align by dates intersection (should match)
        idx = df.index
        # Build mapping from old td.dates to new idx if needed
        dates = pd.DatetimeIndex(idx)
        # X must align too
        X = df[list(ALL_FEATURE_COLS)].values.astype(np.float32)
        # Do NOT hard-clip here. Winsorization is applied later using train-only bounds.
        y_m_raw = df[target_col].values.astype(np.float32)
        y_m = y_m_raw.copy()
        y_d = df["Daily_Return"].values.astype(np.float32)
        y_d_raw = y_d.copy()
        r = df["HMM_Regime"].values.astype(np.int64)
        patched[t] = TickerData(
            ticker=t,
            ticker_id=td.ticker_id,
            dates=dates,
            X=X,
            y_monthly=y_m,
            y_daily=y_d,
            y_monthly_raw=y_m_raw,
            y_daily_raw=y_d_raw,
            regime=r,
        )
    data = patched

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    train_keys: list[tuple[str, int]] = []
    for _t, td in data.items():
        train_keys.extend(_build_sample_keys(td, train_start_ts, train_end_ts + pd.Timedelta(days=1), SEQ_LEN))

    if len(train_keys) < 5000:
        raise ValueError(f"Not enough training samples: {len(train_keys)}")

    # Validation window = last ~3 rebalance steps worth of days from the training period
    rebal_train = _rebalance_dates_21d("data/raw/_NSEI.parquet", train_start, train_end, step_size=step_size)
    if len(rebal_train) >= 4:
        val_start = pd.Timestamp(rebal_train[-3])
    else:
        val_start = train_end_ts - pd.Timedelta(days=120)

    train_keys_final: list[tuple[str, int]] = []
    val_keys: list[tuple[str, int]] = []
    for t, td in data.items():
        train_keys_final.extend(_build_sample_keys(td, train_start_ts, val_start, SEQ_LEN))
        val_keys.extend(_build_sample_keys(td, val_start, train_end_ts + pd.Timedelta(days=1), SEQ_LEN))

    scaler = _fit_scaler_on_train(data, train_keys_final)
    data_sc = _apply_scaler(data, scaler)

    # Winsorization bounds from TRAIN ONLY (1st/99th percentiles) on raw monthly labels
    y_train_raw = []
    for t, i in train_keys_final:
        y_train_raw.append(float(data_sc[t].y_monthly_raw[i]))
    y_train_raw = np.asarray(y_train_raw, dtype=np.float32)
    lo_b = float(np.nanpercentile(y_train_raw, 1.0))
    hi_b = float(np.nanpercentile(y_train_raw, 99.0))

    # Apply winsorization to all tickers using train-derived bounds
    patched_w: dict[str, TickerData] = {}
    for t, td in data_sc.items():
        y_raw_w = _winsorize_with_bounds(td.y_monthly_raw, lo_b, hi_b)
        patched_w[t] = TickerData(
            ticker=td.ticker,
            ticker_id=td.ticker_id,
            dates=td.dates,
            X=td.X,
            y_monthly=y_raw_w.copy(),
            y_daily=td.y_daily,
            y_monthly_raw=y_raw_w,
            y_daily_raw=td.y_daily_raw,
            regime=td.regime,
        )
    data_sc = patched_w

    # Label scaler (monthly target) fit on training only (after winsorization)
    y_scaler = _fit_y_scaler_on_train(data_sc, train_keys_final)
    data_sc = _apply_y_scaler(data_sc, y_scaler)

    train_ds = MultiTickerSequenceDataset(data_sc, sorted(train_keys_final), SEQ_LEN)
    val_ds = MultiTickerSequenceDataset(data_sc, sorted(val_keys), SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2)

    model = build_ramt({"seq_len": SEQ_LEN, "num_heads": NUM_HEADS, "dropout": MODEL_DROPOUT}).to(DEVICE)
    criterion = CombinedLoss(lambda_dir=LAMBDA_DIR)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best = float("inf")
    best_state = None
    patience = 0
    epochs = MAX_EPOCHS if max_epochs is None else int(max_epochs)

    print(
        f"\nFixed-train: train<{train_start_ts.date()}→{train_end_ts.date()}> "
        f"val_start<{val_start.date()}> "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)}",
        flush=True,
    )

    for epoch in range(epochs):
        _ = _train_one_epoch(model, train_loader, optimizer, criterion)
        v = _eval_loss(model, val_loader, criterion)
        if v < best:
            best = v
            patience = 0
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        else:
            patience += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:02d}/{epochs} val_loss={v:.6f}", flush=True)
        if patience >= PATIENCE:
            break
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save artifacts for later inspection (e.g., attention maps)
    save_ramt_inference_artifacts(
        Path("results"),
        model=model,
        scaler=scaler,
        y_scaler=y_scaler,
        train_start=str(train_start_ts.date()),
        train_end=str(train_end_ts.date()),
        lo_b=lo_b,
        hi_b=hi_b,
        fold_label=f"fixed_train_{train_start_ts.date()}_{train_end_ts.date()}",
        fold_tag="fixed_train_latest",
    )

    # Predict on rebalance dates in test period
    rebal_test = _rebalance_dates_21d("data/raw/_NSEI.parquet", test_start, test_end, step_size=step_size)
    rows: list[dict[str, object]] = []
    for t, td in data_sc.items():
        for d in rebal_test:
            if d < test_start_ts or d > test_end_ts:
                continue
            try:
                i = int(td.dates.get_loc(pd.Timestamp(d)))
            except KeyError:
                continue
            if i < SEQ_LEN:
                continue
            Xseq = torch.from_numpy(td.X[i - SEQ_LEN : i]).float().unsqueeze(0).to(DEVICE)
            r = torch.tensor([int(td.regime[i])], dtype=torch.long).to(DEVICE)
            tid = torch.tensor([int(td.ticker_id)], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                pred_m_sc, _pred_d, _g = model(Xseq, r, ticker_id=tid)
            # inverse-transform monthly prediction back to raw alpha units
            pred_m = float(y_scaler.inverse_transform([[float(pred_m_sc.cpu().numpy().squeeze())]])[0][0])
            rows.append(
                {
                    "Date": pd.Timestamp(d),
                    "Ticker": t,
                    "predicted_alpha": float(pred_m),
                    "actual_alpha": float(td.y_monthly_raw[i]),
                    "fold_train_end": train_end_ts,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "predicted_alpha", "actual_alpha", "fold_train_end"])
    return pd.DataFrame(rows).sort_values(["Date", "predicted_alpha"], ascending=[True, False])


if __name__ == "__main__":
    print("Combined ranking training (walk-forward).", flush=True)
    df = combined_walk_forward()
    os.makedirs("results", exist_ok=True)
    out = "results/ranking_predictions.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
