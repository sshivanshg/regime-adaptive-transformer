"""
RAMT Baseline Model — XGBoost
Walk-forward validation with expanding training window.
Evaluates RMSE, MAE, Directional Accuracy, and Sharpe Ratio
on out-of-sample predictions for all 4 tickers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

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
VAL_FRAC_WITHIN_TRAIN = 0.2
MIN_REGIME_SAMPLES = 50
TOP_N_FEATURES = 15
REGIME_CODES = (0, 1, 2)

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Target = next-day log return; align rows where y is defined."""
    df = df.sort_values("Date").reset_index(drop=True)
    y = df["Log_Return"].shift(-1)
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    dates = df["Date"]
    regime = df["HMM_Regime"].fillna(-1).astype(int)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    X = df[feat_cols]
    return X, y, dates, regime


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """1.5x weight when |y| > 1 std of y; else 1.0."""
    ys = y.to_numpy(dtype=float)
    std = float(np.std(ys, ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return np.ones(len(ys), dtype=float)
    return np.where(np.abs(ys) > std, 1.5, 1.0).astype(float)


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    n0: int,
) -> list[str]:
    """Fit on initial window (with same train/val split as folds) and return top n feature names."""
    n_val = max(1, int(np.ceil(VAL_FRAC_WITHIN_TRAIN * n0)))
    n_fit = n0 - n_val
    if n_fit < 10:
        raise ValueError("Initial window too small for feature selection")
    X_init = X.iloc[:n0]
    y_init = y.iloc[:n0]
    X_fit = X_init.iloc[:n_fit]
    y_fit = y_init.iloc[:n_fit]
    X_val = X_init.iloc[n_fit:]
    y_val = y_init.iloc[n_fit:]
    w_fit = compute_sample_weights(y_fit)
    fs_model = XGBRegressor(**XGB_PARAMS)
    fs_model.fit(
        X_fit,
        y_fit,
        sample_weight=w_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    names = list(X_fit.columns)
    imp = fs_model.feature_importances_
    order = np.argsort(imp)[::-1][:TOP_N_FEATURES]
    return [names[i] for i in order]


def cols_without_regime(selected: list[str]) -> list[str]:
    return [c for c in selected if c != "HMM_Regime"]


def _fit_regime_model(
    X_fit_r: pd.DataFrame,
    y_fit_r: pd.Series,
    X_val_r: pd.DataFrame,
    y_val_r: pd.Series,
) -> XGBRegressor:
    w_fit_r = compute_sample_weights(y_fit_r)
    model = XGBRegressor(**XGB_PARAMS)
    if len(X_val_r) >= 1:
        model.fit(
            X_fit_r,
            y_fit_r,
            sample_weight=w_fit_r,
            eval_set=[(X_val_r, y_val_r)],
            verbose=False,
        )
    else:
        nvr = len(X_fit_r)
        if nvr < 20:
            model.fit(X_fit_r, y_fit_r, sample_weight=w_fit_r, verbose=False)
        else:
            nv2 = max(1, int(np.ceil(0.2 * nvr)))
            model.fit(
                X_fit_r.iloc[:-nv2],
                y_fit_r.iloc[:-nv2],
                sample_weight=w_fit_r[:-nv2],
                eval_set=[(X_fit_r.iloc[-nv2:], y_fit_r.iloc[-nv2:])],
                verbose=False,
            )
    return model


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    regime: pd.Series,
    ticker: str,
    selected_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expanding window with regime-stratified models per fold: train up to 3
    XGBRegressors (HMM regimes 0,1,2) plus a global fallback; route test
    predictions by row regime when the regime had >= MIN_REGIME_SAMPLES train rows.
    """
    cols_no_reg = cols_without_regime(selected_features)

    n = len(X)
    n0 = int(n * INITIAL_TRAIN_FRAC)
    if n0 < 50 or n0 + TEST_DAYS > n:
        raise ValueError(f"{ticker}: insufficient rows (n={n}, n0={n0})")

    train_end_probe = n0
    total_folds = 0
    while train_end_probe + TEST_DAYS <= n:
        total_folds += 1
        train_end_probe += TEST_DAYS

    print(f"WALK-FORWARD {ticker}")
    print(f"Initial training size: {n0}")
    print(f"Step size (train_end advance): {TEST_DAYS} trading days")
    print(f"Test window per fold: {TEST_DAYS} trading days")
    print(f"STEP_DAYS constant: {STEP_DAYS} (must match step if used elsewhere)")
    print(f"Total folds: {total_folds}")
    print(f"XGB random_state: {XGB_PARAMS.get('random_state')}")

    oos_dates: list = []
    oos_y_true: list = []
    oos_y_pred: list = []

    train_end = n0
    while train_end + TEST_DAYS <= n:
        test_start = train_end
        test_end = train_end + TEST_DAYS

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        regime_tr = regime.iloc[:train_end]

        n_tr = len(X_tr)
        n_val = max(1, int(np.ceil(VAL_FRAC_WITHIN_TRAIN * n_tr)))
        n_fit = n_tr - n_val
        if n_fit < 10:
            raise ValueError(f"{ticker}: training split too small at train_end={train_end}")

        X_fit = X_tr.iloc[:n_fit]
        y_fit = y_tr.iloc[:n_fit]
        X_val = X_tr.iloc[n_fit:]
        y_val = y_tr.iloc[n_fit:]
        regime_fit = regime_tr.iloc[:n_fit]
        regime_val = regime_tr.iloc[n_fit:]

        w_fit = compute_sample_weights(y_fit)
        global_model = XGBRegressor(**XGB_PARAMS)
        global_model.fit(
            X_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        regime_models: dict[int, XGBRegressor] = {}
        use_regime: dict[int, bool] = {r: False for r in REGIME_CODES}

        for r in REGIME_CODES:
            mask_fit = regime_fit == r
            n_r = int(mask_fit.sum())
            if n_r < MIN_REGIME_SAMPLES:
                continue
            X_fit_r = X_fit.loc[mask_fit, cols_no_reg]
            y_fit_r = y_fit.loc[mask_fit]
            mask_val = regime_val == r
            X_val_r = X_val.loc[mask_val, cols_no_reg]
            y_val_r = y_val.loc[mask_val]
            regime_models[r] = _fit_regime_model(X_fit_r, y_fit_r, X_val_r, y_val_r)
            use_regime[r] = True

        X_te = X.iloc[test_start:test_end]
        y_te = y.iloc[test_start:test_end]
        d_te = dates.iloc[test_start:test_end]
        regime_te = regime.iloc[test_start:test_end]

        preds: list[float] = []
        for j in range(len(X_te)):
            r_i = int(regime_te.iloc[j])
            row = X_te.iloc[j : j + 1]
            if r_i in use_regime and use_regime[r_i]:
                preds.append(float(regime_models[r_i].predict(row[cols_no_reg])[0]))
            else:
                preds.append(float(global_model.predict(row[selected_features])[0]))

        oos_dates.extend(d_te.tolist())
        oos_y_true.extend(y_te.to_numpy().tolist())
        oos_y_pred.extend(preds)

        train_end += TEST_DAYS

    return (
        np.array(oos_dates, dtype="datetime64[ns]"),
        np.array(oos_y_true, dtype=float),
        np.array(oos_y_pred, dtype=float),
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(mean_absolute_error(a, b))


def directional_accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    st = np.sign(y_true)
    sp = np.sign(y_pred)
    return float(np.mean(st == sp) * 100.0)


def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Soft position: y_pred scaled by rolling std of predictions, clipped to [-2, 2]."""
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

    all_rows: list[dict] = []
    metrics_rows: list[tuple[str, float, float, float, float]] = []
    top_features_by_ticker: dict[str, list[str]] = {}

    for ticker, fname in FEATURE_FILES:
        raw = data_raw[ticker]
        X, y, dates, regime = prepare_xy(raw)
        n = len(X)
        n0 = int(n * INITIAL_TRAIN_FRAC)
        selected = select_top_features(X, y, n0)
        top_features_by_ticker[ticker] = selected
        print("=== FEATURES USED ===")
        print(f"{ticker} — number of features: {len(selected)}")
        print(f"Feature columns: {selected}")
        X = X[selected]

        dates_oos, y_true, y_pred = walk_forward_predict(
            X, y, dates, regime, ticker, selected
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
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    out_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "xgboost_predictions.csv"
    out_df.to_csv(out_path, index=False)

    print()
    print(f"Saved predictions → {out_path.resolve()}")
    print()
    print("Table 1 — Regime-stratified results:")
    print("Ticker       | RMSE   | MAE    | DA%   | Sharpe")
    print("-------------|--------|--------|-------|-------")

    rmses, maes, das, shs = [], [], [], []
    for ticker, r, m, da, sh in metrics_rows:
        rmses.append(r)
        maes.append(m)
        das.append(da)
        shs.append(sh if not np.isnan(sh) else np.nan)
        sh_str = f"{sh:>6.2f}" if not np.isnan(sh) else "   nan"
        print(
            f"{ticker:<12} | {r:.4f} | {m:.4f} | {da:>5.2f} | {sh_str}"
        )

    avg_r = float(np.nanmean(rmses))
    avg_m = float(np.nanmean(maes))
    avg_da = float(np.nanmean(das))
    avg_sh = float(np.nanmean([s for s in shs if not np.isnan(s)]))
    if np.all(np.isnan(shs)):
        avg_sh_str = "   nan"
    else:
        avg_sh_str = f"{avg_sh:>6.2f}"
    print(
        f"{'Average':<12} | {avg_r:.4f} | {avg_m:.4f} | {avg_da:>5.2f} | {avg_sh_str}"
    )
    print()
    print("Table 2 — Top 5 features per ticker:")
    w = 22
    hdr = " | ".join([f"{'Rank ' + str(i + 1):<{w}}" for i in range(5)])
    print(f"{'Ticker':<12} | {hdr}")
    print("-" * 12 + "-+-" + "-+-".join(["-" * w] * 5))
    for ticker, _fname in FEATURE_FILES:
        feats = top_features_by_ticker[ticker][:5]
        while len(feats) < 5:
            feats.append("")
        cols = [f"{f:<{w}}" for f in feats]
        print(f"{ticker:<12} | " + " | ".join(cols))
    print()


def last_timestep(X: np.ndarray) -> np.ndarray:
    """Legacy helper for sequence tensors: take last timestep."""
    return X[:, -1, :]


def evaluate_classifier(clf: XGBClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """Legacy classifier metrics for `evaluate.py` + saved XGBoost checkpoints."""
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
    }


if __name__ == "__main__":
    main()
