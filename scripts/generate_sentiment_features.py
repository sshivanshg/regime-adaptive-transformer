from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.lora_experiment.chronos_lora import ChronosLoRARanker


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_ticker(v: str) -> str:
    t = str(v).strip().upper().replace(".", "_")
    if t.endswith("_NS"):
        return t
    if t.endswith("NS") and not t.endswith("_NS"):
        return t[:-2] + "_NS"
    return t


def _load_universe(path: Path) -> set[str]:
    if not path.exists():
        return set()
    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(_normalize_ticker(s))
    return set(tickers)


def _resolve_text_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            if df[col].astype(str).str.len().gt(0).any():
                return col
    raise ValueError(f"No usable text column found. Tried: {list(candidates)}")


def _device(name: str) -> torch.device:
    d = str(name).lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_chronos_lora(input_dim: int, ckpt_path: Path | None, device: torch.device) -> ChronosLoRARanker:
    model = ChronosLoRARanker(input_dim=input_dim)
    if ckpt_path is not None and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict):
            lora_sd = payload.get("encoder_lora_state_dict")
            if lora_sd:
                model.encoder.load_state_dict(lora_sd, strict=False)
            proj_sd = payload.get("input_projection_state_dict")
            if proj_sd:
                model.input_projection.load_state_dict(proj_sd, strict=False)
            head_sd = payload.get("ranking_head_state_dict")
            if head_sd:
                model.ranking_head.load_state_dict(head_sd, strict=False)
    model.eval()
    return model.to(device)


def _batched_finbert(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    scores: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)

            # FinBERT class order: [negative, neutral, positive]
            p_neg = probs[:, 0]
            p_pos = probs[:, 2]
            score = (p_pos - p_neg).clamp(-1.0, 1.0)
            conf = probs.max(dim=-1).values

            scores.append(score.detach().cpu().numpy())
            confs.append(conf.detach().cpu().numpy())

    return np.concatenate(scores), np.concatenate(confs)


def _chronos_calibration(
    daily_df: pd.DataFrame,
    seq_len: int,
    input_dim: int,
    model: ChronosLoRARanker,
    device: torch.device,
) -> pd.Series:
    # Build compact numeric sequence per ticker/date and calibrate to [-1, 1].
    feat_cols = [
        "sentiment_score",
        "sentiment_confidence",
        "mention_count",
        "score_std",
        "score_min",
        "score_max",
    ]
    x = daily_df.copy().sort_values(["Ticker", "Date"])
    for c in feat_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    # Pad to match Chronos input dim.
    for i in range(input_dim - len(feat_cols)):
        x[f"aux_{i}"] = 0.0
    model_cols = feat_cols + [f"aux_{i}" for i in range(max(0, input_dim - len(feat_cols)))]

    out = np.zeros(len(x), dtype=np.float32)
    idx = x.index.to_numpy()
    idx_map = {int(v): i for i, v in enumerate(idx)}

    with torch.no_grad():
        for _, g in x.groupby("Ticker", sort=False):
            arr = g[model_cols].to_numpy(dtype=np.float32)
            if len(arr) < seq_len:
                continue
            for j in range(seq_len, len(arr) + 1):
                window = arr[j - seq_len : j]
                t = torch.from_numpy(window).unsqueeze(0).to(device)
                y = model(t).squeeze().detach().cpu().item()
                out_pos = g.index[j - 1]
                out[idx_map[int(out_pos)]] = float(np.tanh(y))

    return pd.Series(out, index=x.index).reindex(daily_df.index).fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily sentiment features per (Date, Ticker).")
    parser.add_argument("--config", type=str, default="config/hybrid_config.yaml")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    paths = cfg.get("paths", {})
    scfg = cfg.get("sentiment_pipeline", {})

    text_path = Path(paths.get("text_input", "data/raw/news_text.parquet"))
    out_path = Path(paths.get("sentiment_output", "data/processed/sentiment_features.parquet"))
    tickers_path = Path(paths.get("nifty200_tickers", "data/nifty200_tickers.txt"))

    if not text_path.exists():
        raise SystemExit(
            f"Missing text input: {text_path}. Provide a parquet with Date/Ticker + text columns."
        )

    df = pd.read_parquet(text_path)
    req = scfg.get("required_columns", {"date": "Date", "ticker": "Ticker"})
    date_col = req.get("date", "Date")
    ticker_col = req.get("ticker", "Ticker")
    if date_col not in df.columns or ticker_col not in df.columns:
        raise ValueError(f"Input must include {date_col} and {ticker_col} columns.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df["Ticker"] = df[ticker_col].map(_normalize_ticker)

    universe = _load_universe(tickers_path)
    if universe:
        df = df[df["Ticker"].isin(universe)].copy()

    text_col = _resolve_text_column(df, scfg.get("text_columns", ["text", "headline"]))
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].copy()
    if df.empty:
        raise SystemExit("No text rows after filtering to universe/non-empty text.")

    device = _device(scfg.get("device", "auto"))

    score, conf = _batched_finbert(
        texts=df[text_col].tolist(),
        model_name=scfg.get("finbert_model_name", "ProsusAI/finbert"),
        batch_size=int(scfg.get("finbert_batch_size", 32)),
        max_length=int(scfg.get("finbert_max_length", 192)),
        device=device,
    )
    df["sentiment_score_text"] = score
    df["sentiment_confidence_text"] = conf

    # Aggregate to daily ticker sentiment (confidence-weighted).
    daily = (
        df.assign(_w=df["sentiment_confidence_text"].clip(lower=0.0))
        .groupby(["Date", "Ticker"], as_index=False)
        .agg(
            mention_count=(text_col, "count"),
            score_mean=("sentiment_score_text", "mean"),
            score_std=("sentiment_score_text", "std"),
            score_min=("sentiment_score_text", "min"),
            score_max=("sentiment_score_text", "max"),
            confidence_mean=("sentiment_confidence_text", "mean"),
            w_sum=("_w", "sum"),
        )
    )
    # Recompute weighted score with vectorized merge for stability.
    wt = (
        df.assign(_num=df["sentiment_score_text"] * df["sentiment_confidence_text"].clip(lower=0.0))
        .groupby(["Date", "Ticker"], as_index=False)
        .agg(_num=("_num", "sum"), _den=("sentiment_confidence_text", lambda x: float(np.clip(x, 0.0, None).sum())))
    )
    daily = daily.merge(wt, on=["Date", "Ticker"], how="left")
    daily["sentiment_score"] = np.where(daily["_den"] > 1e-12, daily["_num"] / daily["_den"], daily["score_mean"])
    daily["sentiment_confidence"] = daily["confidence_mean"].clip(0.0, 1.0)

    if bool(scfg.get("use_chronos_calibration", True)):
        ckpt = Path(paths.get("lora_adapter_checkpoint", ""))
        chronos = _load_chronos_lora(
            input_dim=int(scfg.get("chronos_input_dim", 10)),
            ckpt_path=ckpt if str(ckpt) else None,
            device=device,
        )
        cal = _chronos_calibration(
            daily_df=daily,
            seq_len=int(scfg.get("chronos_seq_len", 20)),
            input_dim=int(scfg.get("chronos_input_dim", 10)),
            model=chronos,
            device=device,
        )
        blend = float(scfg.get("chronos_blend_weight_finbert", 0.8))
        daily["sentiment_score"] = np.clip(
            blend * daily["sentiment_score"].astype(float) + (1.0 - blend) * cal.astype(float),
            -1.0,
            1.0,
        )

    out = daily[["Date", "Ticker", "sentiment_score", "sentiment_confidence"]].copy()
    out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(
        f"Wrote sentiment features: {out_path} | rows={len(out):,} "
        f"dates={out['Date'].nunique():,} tickers={out['Ticker'].nunique():,}"
    )


if __name__ == "__main__":
    main()
