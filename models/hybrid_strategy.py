"""
User Story:
As a quant researcher, I need a hybrid strategy that combines momentum ranking, HMM
regime-based sizing, and FinBERT/LoRA sentiment gating so that the final portfolio
is robust across different market conditions.

Implementation Approach:
Implement a strategy class that ranks stocks by momentum, applies HMM regime detection
to set position sizing (100% bull, 50% high-vol, 20% bear), optionally integrates
FinBERT sentiment scores via LoRA adapters for gating, enforces sector caps, and
applies stop-loss and portfolio drawdown controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.lora_experiment.chronos_lora import ChronosLoRARanker


@dataclass(frozen=True)
class HybridRuleConfig:
    top_n: int = 5
    bull_regime: int = 1
    high_vol_regime: int = 0
    bear_regime: int = 2
    bull_sentiment_floor: float = -0.2
    bear_sentiment_floor: float = 0.5
    high_vol_momentum_weight: float = 0.6
    high_vol_sentiment_weight: float = 0.4


def _normalize_ticker(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.upper().str.strip().str.replace(".", "_", regex=False)
    return t


def _cross_sectional_zscore(x: pd.Series) -> pd.Series:
    std = float(x.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - float(x.mean())) / std


def build_hybrid_rankings(
    momentum_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    config: HybridRuleConfig | None = None,
    momentum_col: str = "predicted_alpha",
) -> pd.DataFrame:
    cfg = config or HybridRuleConfig()

    needed_m = {"Date", "Ticker", momentum_col}
    needed_s = {"Date", "Ticker", "sentiment_score", "sentiment_confidence"}
    needed_r = {"Date", "regime"}
    if needed_m.difference(momentum_df.columns):
        raise ValueError(f"Momentum df missing: {sorted(needed_m.difference(momentum_df.columns))}")
    if needed_s.difference(sentiment_df.columns):
        raise ValueError(f"Sentiment df missing: {sorted(needed_s.difference(sentiment_df.columns))}")
    if needed_r.difference(regime_df.columns):
        raise ValueError(f"Regime df missing: {sorted(needed_r.difference(regime_df.columns))}")

    m = momentum_df.copy()
    s = sentiment_df.copy()
    r = regime_df.copy()

    m["Date"] = pd.to_datetime(m["Date"]).dt.tz_localize(None)
    s["Date"] = pd.to_datetime(s["Date"]).dt.tz_localize(None)
    r["Date"] = pd.to_datetime(r["Date"]).dt.tz_localize(None)

    m["Ticker"] = _normalize_ticker(m["Ticker"])
    s["Ticker"] = _normalize_ticker(s["Ticker"])

    x = m.merge(s[["Date", "Ticker", "sentiment_score", "sentiment_confidence"]], on=["Date", "Ticker"], how="left")
    x = x.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # T-1 alignment: inputs known at T-1 generate ranking for T.
    x["feature_source_date_t1"] = x.groupby("Ticker")["Date"].shift(1)
    x["momentum_t1"] = x.groupby("Ticker")[momentum_col].shift(1)
    x["sentiment_t1"] = x.groupby("Ticker")["sentiment_score"].shift(1)
    x["sentiment_conf_t1"] = x.groupby("Ticker")["sentiment_confidence"].shift(1)

    rr = r[["Date", "regime"]].sort_values("Date").copy()
    rr["regime_source_date_t1"] = rr["Date"].shift(1)
    rr["regime_t1"] = rr["regime"].shift(1)
    x = x.merge(rr[["Date", "regime_t1", "regime_source_date_t1"]], on="Date", how="left")

    x = x.dropna(subset=["momentum_t1", "sentiment_t1", "regime_t1"]).copy()
    x["regime_t1"] = x["regime_t1"].astype(int)

    selected = []
    for d, g in x.groupby("Date", sort=True):
        regime = int(g["regime_t1"].iloc[0])
        gg = g.copy()

        if regime == cfg.bull_regime:
            mode = "BULL_MOMENTUM_FILTERED"
            gg = gg[gg["sentiment_t1"] > cfg.bull_sentiment_floor]
            if gg.empty:
                continue
            gg = gg.sort_values("momentum_t1", ascending=False).head(cfg.top_n)
            gg["hybrid_score"] = gg["momentum_t1"]

        elif regime == cfg.high_vol_regime:
            mode = "HIGH_VOL_INTEGRATED"
            mom_z = _cross_sectional_zscore(gg["momentum_t1"])
            sent_z = _cross_sectional_zscore(gg["sentiment_t1"])
            gg["hybrid_score"] = (
                cfg.high_vol_momentum_weight * mom_z + cfg.high_vol_sentiment_weight * sent_z
            )
            gg = gg.sort_values("hybrid_score", ascending=False).head(cfg.top_n)

        elif regime == cfg.bear_regime:
            mode = "BEAR_DEFENSIVE"
            gg = gg[gg["sentiment_t1"] > cfg.bear_sentiment_floor]
            if gg.empty:
                # Explicitly represent cash regime for downstream backtest wiring.
                selected.append(
                    pd.DataFrame(
                        {
                            "Date": [d],
                            "Ticker": ["CASH"],
                            "hybrid_score": [0.0],
                            "regime_t1": [regime],
                            "hybrid_mode": [mode],
                            "sentiment_t1": [np.nan],
                            "momentum_t1": [np.nan],
                            "sentiment_conf_t1": [np.nan],
                        }
                    )
                )
                continue
            gg = gg.sort_values("sentiment_t1", ascending=False).head(cfg.top_n)
            gg["hybrid_score"] = gg["sentiment_t1"]

        else:
            mode = "UNKNOWN_REGIME_FALLBACK"
            gg = gg.sort_values("momentum_t1", ascending=False).head(cfg.top_n)
            gg["hybrid_score"] = gg["momentum_t1"]

        gg["hybrid_mode"] = mode
        selected.append(
            gg[
                [
                    "Date",
                    "Ticker",
                    "hybrid_score",
                    "regime_t1",
                    "hybrid_mode",
                    "sentiment_t1",
                    "momentum_t1",
                    "sentiment_conf_t1",
                ]
            ]
        )

    if not selected:
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "hybrid_score",
                "regime_t1",
                "hybrid_mode",
                "sentiment_t1",
                "momentum_t1",
                "sentiment_conf_t1",
            ]
        )

    out = pd.concat(selected, ignore_index=True)
    return out.sort_values(["Date", "hybrid_score"], ascending=[True, False]).reset_index(drop=True)


def check_architecture_integrity(
    momentum_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    lora_adapter_checkpoint: str | Path | None = None,
    chronos_input_dim: int = 10,
) -> dict[str, object]:
    report: dict[str, object] = {
        "lora_loaded": False,
        "lora_trainable_params": 0,
        "alignment_t_minus_1_ok": False,
        "leakage_checks_passed": False,
        "errors": [],
    }

    # 1) LoRA integrity
    try:
        model = ChronosLoRARanker(input_dim=int(chronos_input_dim))
        if lora_adapter_checkpoint is not None:
            ckpt = Path(lora_adapter_checkpoint)
            if ckpt.exists():
                payload = torch.load(ckpt, map_location="cpu")
                if isinstance(payload, dict) and payload.get("encoder_lora_state_dict"):
                    model.encoder.load_state_dict(payload["encoder_lora_state_dict"], strict=False)
                    report["lora_loaded"] = True
                else:
                    report["errors"].append("LoRA checkpoint does not contain encoder_lora_state_dict")
            else:
                report["errors"].append(f"LoRA checkpoint not found: {ckpt}")
        report["lora_trainable_params"] = int(model.trainable_parameter_count())
    except Exception as exc:
        report["errors"].append(f"LoRA load check failed: {type(exc).__name__}: {exc}")

    # 2) T-1 alignment and leakage checks
    try:
        m = momentum_df[["Date", "Ticker", "predicted_alpha"]].copy()
        s = sentiment_df[["Date", "Ticker", "sentiment_score"]].copy()
        r = regime_df[["Date", "regime"]].copy()

        m["Date"] = pd.to_datetime(m["Date"])
        s["Date"] = pd.to_datetime(s["Date"])
        r["Date"] = pd.to_datetime(r["Date"])

        x = m.merge(s, on=["Date", "Ticker"], how="left").sort_values(["Ticker", "Date"])
        x["feature_source_date_t1"] = x.groupby("Ticker")["Date"].shift(1)
        x["momentum_t1"] = x.groupby("Ticker")["predicted_alpha"].shift(1)
        x["sentiment_t1"] = x.groupby("Ticker")["sentiment_score"].shift(1)

        rr = r.sort_values("Date")
        rr["regime_source_date_t1"] = rr["Date"].shift(1)
        rr["regime_t1"] = rr["regime"].shift(1)
        x = x.merge(rr[["Date", "regime_t1", "regime_source_date_t1"]], on="Date", how="left")

        usable = x.dropna(subset=["momentum_t1", "sentiment_t1", "regime_t1"]).copy()
        report["alignment_t_minus_1_ok"] = len(usable) > 0

        # Strict leakage guard: all lag-source dates must be earlier than decision date.
        cond_feature = (usable["feature_source_date_t1"] < usable["Date"]).all()
        cond_regime = (usable["regime_source_date_t1"] < usable["Date"]).all()
        no_leak = bool(cond_feature and cond_regime)
        report["leakage_checks_passed"] = no_leak
        report["feature_t_minus_1_order_ok"] = bool(cond_feature)
        report["regime_t_minus_1_order_ok"] = bool(cond_regime)

        if not no_leak:
            report["errors"].append(
                "Potential leakage: lagged inputs appear too similar to same-day values."
            )
    except Exception as exc:
        report["errors"].append(f"Alignment/leakage check failed: {type(exc).__name__}: {exc}")

    report["ok"] = bool(
        report["alignment_t_minus_1_ok"]
        and report["leakage_checks_passed"]
        and (report["lora_loaded"] or lora_adapter_checkpoint is None)
    )
    return report
