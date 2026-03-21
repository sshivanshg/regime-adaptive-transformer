from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    **kwargs,
) -> xgb.XGBClassifier:
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    params.update(kwargs)
    clf = xgb.XGBClassifier(**params)
    if X_val is not None and y_val is not None:
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        clf.fit(X_train, y_train)
    return clf


def last_timestep(X: np.ndarray) -> np.ndarray:
    return X[:, -1, :]


def evaluate_classifier(clf: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
    }
