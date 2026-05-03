"""
User Story:
Generate reproducible daily sentiment features from raw text for NIFTY 200 tickers,
with optional LoRA-calibrated scoring and cache-aware inference.

Implementation Approach:
Map text rows to ticker/date keys, reuse cached per-item predictions when possible,
run batched FinBERT inference with memory-safe settings, optionally blend with
Chronos LoRA calibration, then aggregate to daily ticker-level sentiment.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.lora_experiment.chronos_lora import ChronosLoRARanker


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline_sentiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


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


def _load_nifty200(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        t = _normalize_ticker(s)
        if t in {"_NSEI", "NSEI", "^NSEI"}:
            continue
        out.add(t)
    return out


def _resolve_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required columns. Candidates tried: {list(candidates)}")
    return None


def _device(name: str) -> torch.device:
    x = str(name).lower()
    if x == "cpu":
        return torch.device("cpu")
    if x == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if x == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_hash(date: pd.Timestamp, ticker: str, text: str, model_tag: str) -> str:
    payload = f"{date.date()}|{ticker}|{model_tag}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        c = pd.read_parquet(cache_path)
        for col in ["cache_key", "sentiment_score", "sentiment_confidence"]:
            if col not in c.columns:
                raise ValueError(f"Cache missing column: {col}")
        return c
    return pd.DataFrame(columns=["cache_key", "sentiment_score", "sentiment_confidence", "model_tag", "updated_at"])


def _save_cache(cache_path: Path, cache_df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dedup = cache_df.drop_duplicates(subset=["cache_key"], keep="last")
    dedup.to_parquet(cache_path, index=False)


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
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)

            # FinBERT order: negative, neutral, positive
            score = (probs[:, 2] - probs[:, 0]).clamp(-1.0, 1.0)
            conf = probs.max(dim=-1).values

            scores.append(score.cpu().numpy())
            confs.append(conf.cpu().numpy())

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return np.concatenate(scores), np.concatenate(confs)


def _apply_lora_calibration(
    finbert_score: np.ndarray,
    finbert_conf: np.ndarray,
    text_len: np.ndarray,
    recency_w: np.ndarray,
    *,
    input_dim: int,
    checkpoint: Path | None,
    device: torch.device,
) -> np.ndarray:
    model = ChronosLoRARanker(input_dim=input_dim).to(device)
    model.eval()

    if checkpoint is not None and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, dict):
            if payload.get("encoder_lora_state_dict"):
                model.encoder.load_state_dict(payload["encoder_lora_state_dict"], strict=False)
            if payload.get("input_projection_state_dict"):
                model.input_projection.load_state_dict(payload["input_projection_state_dict"], strict=False)
            if payload.get("ranking_head_state_dict"):
                model.ranking_head.load_state_dict(payload["ranking_head_state_dict"], strict=False)

    z = np.zeros((len(finbert_score), input_dim), dtype=np.float32)
    z[:, 0] = finbert_score.astype(np.float32)
    z[:, 1] = finbert_conf.astype(np.float32)
    z[:, 2] = np.clip(text_len.astype(np.float32) / 1024.0, 0.0, 1.0)
    z[:, 3] = recency_w.astype(np.float32)

    with torch.no_grad():
        x = torch.from_numpy(z).unsqueeze(1).to(device)  # (batch, seq=1, dim)
        y = model(x).detach().cpu().numpy()
    out = np.tanh(y.astype(np.float32))
    return np.clip(out, -1.0, 1.0)


def _aggregate_daily(news_scored: pd.DataFrame, half_life_hours: float) -> pd.DataFrame:
    out = news_scored.copy()
    age_hours = pd.to_numeric(out["age_hours"], errors="coerce").fillna(0.0)
    recency = np.exp(-np.log(2.0) * np.clip(age_hours, 0.0, None) / max(half_life_hours, 1e-6))

    conf = np.clip(pd.to_numeric(out["sentiment_confidence"], errors="coerce").fillna(0.0), 0.0, 1.0)
    w = recency * np.maximum(conf, 1e-3)
    out["_w"] = w
    out["_num"] = out["sentiment_score"] * out["_w"]

    daily = (
        out.groupby(["Date", "Ticker"], as_index=False)
        .agg(
            sentiment_score_num=("_num", "sum"),
            sentiment_weight_den=("_w", "sum"),
            sentiment_confidence=("sentiment_confidence", "mean"),
            mention_count=("sentiment_score", "count"),
        )
    )
    daily["sentiment_score"] = np.where(
        daily["sentiment_weight_den"] > 1e-12,
        daily["sentiment_score_num"] / daily["sentiment_weight_den"],
        0.0,
    )
    daily["sentiment_score"] = np.clip(daily["sentiment_score"], -1.0, 1.0)
    daily["sentiment_confidence"] = np.clip(daily["sentiment_confidence"], 0.0, 1.0)

    return daily[["Date", "Ticker", "sentiment_score", "sentiment_confidence", "mention_count"]]


def run_pipeline(config_path: Path, use_lora_adapters: bool) -> Path:
    cfg = _load_yaml(config_path)
    paths = cfg.get("paths", {})
    scfg = cfg.get("sentiment_pipeline", {})

    log_path = Path(paths.get("pipeline_log", "logs/pipeline.log"))
    logger = _setup_logger(log_path)

    text_path = Path(paths.get("text_input", "data/raw/news_text.parquet"))
    sentiment_dir = Path(paths.get("sentiment_cache_dir", "data/processed/sentiment"))
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    out_name = "sentiment_features_lora.parquet" if use_lora_adapters else "sentiment_features_vanilla.parquet"
    out_path = sentiment_dir / out_name
    cache_path = sentiment_dir / "inference_cache.parquet"

    tickers_path = Path(paths.get("nifty200_tickers", "data/nifty200_tickers.txt"))
    universe = _load_nifty200(tickers_path)

    if not text_path.exists():
        raise SystemExit(f"Missing text input file: {text_path}")

    logger.info("Loading text input: %s", text_path)
    news = pd.read_parquet(text_path)

    date_col = _resolve_col(news, scfg.get("date_columns", ["Date", "date", "published_at"]))
    ticker_col = _resolve_col(news, scfg.get("ticker_columns", ["Ticker", "ticker", "symbol"]))
    text_col = _resolve_col(news, scfg.get("text_columns", ["text", "headline", "summary", "content"]))
    ts_col = _resolve_col(news, scfg.get("timestamp_columns", ["published_at", "timestamp", "Date"]), required=False)

    news = news.copy()
    news["Date"] = pd.to_datetime(news[date_col]).dt.tz_localize(None).dt.floor("D")
    news["Ticker"] = news[ticker_col].map(_normalize_ticker)
    news["text"] = news[text_col].fillna("").astype(str).str.strip()
    if ts_col is not None:
        news["published_at"] = pd.to_datetime(news[ts_col], errors="coerce").dt.tz_localize(None)
    else:
        news["published_at"] = news["Date"]

    news = news[news["text"] != ""].copy()
    if universe:
        in_news = set(news["Ticker"].unique())
        skipped = sorted(in_news.difference(universe))
        before = len(news)
        news = news[news["Ticker"].isin(universe)].copy()
        logger.info("Filtered to NIFTY200 tickers: %d -> %d rows", before, len(news))
        if skipped:
            logger.info("Skipped %d non-universe tickers during mapping.", len(skipped))

    if news.empty:
        raise SystemExit("No valid text rows after filtering.")

    news["age_hours"] = (
        (news["Date"] + pd.Timedelta(hours=23, minutes=59) - news["published_at"]) / pd.Timedelta(hours=1)
    )
    news["age_hours"] = pd.to_numeric(news["age_hours"], errors="coerce").fillna(0.0).clip(lower=0.0)

    model_tag = "finbert+lora" if use_lora_adapters else "finbert-vanilla"
    news["cache_key"] = [
        _batch_hash(d, t, x, model_tag)
        for d, t, x in zip(news["Date"], news["Ticker"], news["text"])
    ]

    cache = _load_cache(cache_path)
    cache_hit = news.merge(cache[["cache_key", "sentiment_score", "sentiment_confidence"]], on="cache_key", how="left")

    missing_mask = cache_hit["sentiment_score"].isna() | cache_hit["sentiment_confidence"].isna()
    need = cache_hit[missing_mask].copy()
    logger.info("Cache hits: %d | cache misses: %d", int((~missing_mask).sum()), int(missing_mask.sum()))

    if not need.empty:
        device = _device(scfg.get("device", "auto"))
        logger.info("Running inference on %d rows using %s", len(need), device)

        score, conf = _batched_finbert(
            texts=need["text"].tolist(),
            model_name=scfg.get("finbert_model_name", "ProsusAI/finbert"),
            batch_size=int(scfg.get("finbert_batch_size", 32)),
            max_length=int(scfg.get("finbert_max_length", 192)),
            device=device,
        )

        if use_lora_adapters:
            lora_ckpt = Path(paths.get("lora_adapter_checkpoint", "results/lora/chronos_lora_adapter.pt"))
            cal = _apply_lora_calibration(
                score,
                conf,
                need["text"].str.len().to_numpy(dtype=float),
                np.exp(-0.05 * need["age_hours"].to_numpy(dtype=float)),
                input_dim=int(scfg.get("chronos_input_dim", 10)),
                checkpoint=lora_ckpt,
                device=device,
            )
            blend = float(scfg.get("chronos_blend_weight_finbert", 0.8))
            score = np.clip(blend * score + (1.0 - blend) * cal, -1.0, 1.0)

        need["sentiment_score"] = score
        need["sentiment_confidence"] = conf

        cache_new = need[["cache_key", "sentiment_score", "sentiment_confidence"]].copy()
        cache_new["model_tag"] = model_tag
        cache_new["updated_at"] = datetime.utcnow().isoformat()
        cache = pd.concat([cache, cache_new], ignore_index=True)
        _save_cache(cache_path, cache)

        cache_hit = cache_hit.drop(columns=["sentiment_score", "sentiment_confidence"], errors="ignore")
        cache_hit = cache_hit.merge(
            cache[["cache_key", "sentiment_score", "sentiment_confidence"]],
            on="cache_key",
            how="left",
        )

    scored = cache_hit[["Date", "Ticker", "published_at", "age_hours", "sentiment_score", "sentiment_confidence"]].copy()
    daily = _aggregate_daily(scored, half_life_hours=float(scfg.get("aggregation_half_life_hours", 12.0)))
    daily = daily.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, index=False)

    logger.info("Wrote sentiment file: %s | rows=%d | dates=%d | tickers=%d", out_path, len(daily), daily["Date"].nunique(), daily["Ticker"].nunique())
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached sentiment pipeline (vanilla or LoRA-calibrated).")
    parser.add_argument("--config", type=str, default="config/hybrid_config.yaml")
    parser.add_argument("--use-lora-adapters", action="store_true")
    args = parser.parse_args()

    out = run_pipeline(Path(args.config), use_lora_adapters=bool(args.use_lora_adapters))
    print(f"Sentiment pipeline complete: {out}")


if __name__ == "__main__":
    main()
