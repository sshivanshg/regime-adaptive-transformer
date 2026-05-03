"""
User Story:
Provide research-grade explainability for sentiment-driven decisions by identifying
which words drive positive/negative predictions and how this changes by regime.

Implementation Approach:
Run Integrated Gradients on a FinBERT classifier (optionally with LoRA adapter),
aggregate token attributions into interpretable tables/charts, and generate a PDF
case-study contrasting one successful and one failed trade.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ExplainConfig:
    model_name: str
    lora_adapter_path: str | None
    text_input: Path
    backtest_csv: Path
    nifty_features: Path
    out_pdf: Path
    max_regime_samples: int = 20
    ig_steps: int = 24


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


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(model_name: str, lora_adapter_path: str | None, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForSequenceClassification.from_pretrained(model_name)
    if lora_adapter_path:
        ap = Path(lora_adapter_path)
        if ap.exists():
            base = PeftModel.from_pretrained(base, str(ap))
    base.to(device)
    base.eval()
    return tok, base


def integrated_gradients_tokens(
    text: str,
    tokenizer,
    model,
    *,
    target_idx: int | None,
    steps: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    emb_layer = model.get_input_embeddings()
    emb = emb_layer(input_ids)
    baseline = torch.zeros_like(emb)

    with torch.no_grad():
        base_out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(base_out.logits, dim=-1).squeeze(0)
    if target_idx is None:
        target_idx = int(torch.argmax(probs).item())

    total_grad = torch.zeros_like(emb)

    for alpha in torch.linspace(0, 1, steps, device=device):
        x = baseline + alpha * (emb - baseline)
        x.requires_grad_(True)

        out = model(inputs_embeds=x, attention_mask=attention_mask)
        logit = out.logits[:, target_idx].sum()

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        logit.backward(retain_graph=False)

        total_grad += x.grad.detach()

    avg_grad = total_grad / float(steps)
    attr = ((emb - baseline) * avg_grad).sum(dim=-1).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    return tokens, attr.detach().cpu().numpy(), probs.detach().cpu().numpy()


def _load_text_rows(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    date_col = "Date" if "Date" in d.columns else ("date" if "date" in d.columns else "published_at")
    ticker_col = "Ticker" if "Ticker" in d.columns else ("ticker" if "ticker" in d.columns else "symbol")
    text_col = None
    for c in ["text", "headline", "summary", "content"]:
        if c in d.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError("No text-like column found in input parquet.")

    d = d.copy()
    d["Date"] = pd.to_datetime(d[date_col]).dt.tz_localize(None).dt.floor("D")
    d["Ticker"] = d[ticker_col].map(_normalize_ticker)
    d["text"] = d[text_col].fillna("").astype(str)
    return d[d["text"].str.len() > 0].copy()


def _regime_map(nifty_features: Path) -> pd.DataFrame:
    n = pd.read_parquet(nifty_features, columns=["Date", "HMM_Regime"])
    n["Date"] = pd.to_datetime(n["Date"]).dt.tz_localize(None)
    n = n.sort_values("Date").rename(columns={"HMM_Regime": "regime"})
    n["regime"] = n["regime"].astype(int)
    return n


def _parse_stocks(cell) -> list[str]:
    if cell is None:
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    try:
        vals = ast.literal_eval(str(cell))
        return [str(x) for x in vals]
    except Exception:
        return []


def _pick_case_trades(bt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    good = bt.sort_values("portfolio_return", ascending=False).iloc[0]
    bad = bt.sort_values("portfolio_return", ascending=True).iloc[0]
    return good, bad


def _best_headline_for_trade(news: pd.DataFrame, trade_date: pd.Timestamp, tickers: Iterable[str]) -> pd.Series | None:
    cands = news[(news["Date"] <= trade_date) & (news["Ticker"].isin(list(tickers)))].copy()
    if cands.empty:
        return None
    cands["len"] = cands["text"].str.len()
    return cands.sort_values(["Date", "len"], ascending=[False, False]).iloc[0]


def _regime_sensitivity_table(news: pd.DataFrame, regimes: pd.DataFrame, tokenizer, model, device: torch.device, steps: int, max_samples: int) -> pd.DataFrame:
    merged = news.merge(regimes, on="Date", how="left").dropna(subset=["regime"]).copy()
    merged["regime"] = merged["regime"].astype(int)

    risk_words = {"risk", "default", "debt", "downgrade", "selloff", "volatility", "loss"}
    growth_words = {"growth", "beat", "upgrade", "profit", "rally", "guidance", "expansion"}

    rows = []
    for reg, g in merged.groupby("regime"):
        sample = g.head(max_samples)
        risk_attr = []
        growth_attr = []
        for txt in sample["text"].tolist():
            toks, attrs, probs = integrated_gradients_tokens(
                txt,
                tokenizer,
                model,
                target_idx=None,
                steps=steps,
                device=device,
            )
            for t, a in zip(toks, attrs):
                tt = t.lower().replace("##", "")
                if tt in risk_words:
                    risk_attr.append(abs(float(a)))
                if tt in growth_words:
                    growth_attr.append(abs(float(a)))

        rows.append(
            {
                "regime": int(reg),
                "risk_word_attribution": float(np.mean(risk_attr)) if risk_attr else 0.0,
                "growth_word_attribution": float(np.mean(growth_attr)) if growth_attr else 0.0,
                "n_samples": int(len(sample)),
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _plot_token_bar(ax, tokens: list[str], attrs: np.ndarray, title: str) -> None:
    k = min(12, len(tokens))
    idx = np.argsort(np.abs(attrs))[-k:]
    t = [tokens[i].replace("##", "") for i in idx]
    a = attrs[idx]
    colors = ["#22c55e" if x >= 0 else "#ef4444" for x in a]
    ax.barh(range(len(t)), a, color=colors)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t)
    ax.set_title(title)
    ax.axvline(0.0, color="#64748b", linewidth=1)


def build_case_study(cfg: ExplainConfig) -> None:
    device = _device()
    tok, model = _load_model(cfg.model_name, cfg.lora_adapter_path, device)

    news = _load_text_rows(cfg.text_input)
    regimes = _regime_map(cfg.nifty_features)

    bt = pd.read_csv(cfg.backtest_csv, parse_dates=["date"])
    good, bad = _pick_case_trades(bt)

    cases = [
        ("Successful trade", good),
        ("Failed trade", bad),
    ]

    regime_tbl = _regime_sensitivity_table(
        news,
        regimes,
        tok,
        model,
        device,
        steps=cfg.ig_steps,
        max_samples=cfg.max_regime_samples,
    )

    cfg.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(cfg.out_pdf) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.01, 0.95, "Sentiment Explainability Case Study", fontsize=18, weight="bold")
        ax.text(0.01, 0.88, f"Model: {cfg.model_name}", fontsize=11)
        ax.text(0.01, 0.84, "Method: Integrated Gradients token attribution", fontsize=11)
        ax.text(0.01, 0.80, "Focus: one successful and one failed trade + regime sensitivity diagnostic", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for label, row in cases:
            tickers = _parse_stocks(row.get("stocks_held"))
            hd = _best_headline_for_trade(news, pd.Timestamp(row["date"]), tickers)
            if hd is None:
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.02, 0.92, f"{label}: {pd.Timestamp(row['date']).date()}", fontsize=16, weight="bold")
                ax.text(0.02, 0.84, "No matching headline found for held tickers.", fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            tokens, attrs, probs = integrated_gradients_tokens(
                hd["text"],
                tok,
                model,
                target_idx=None,
                steps=cfg.ig_steps,
                device=device,
            )

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1, 1.4]})
            ax1.axis("off")
            ax1.text(0.01, 0.92, f"{label}: {pd.Timestamp(row['date']).date()}", fontsize=15, weight="bold")
            ax1.text(0.01, 0.82, f"Ticker: {hd['Ticker']}", fontsize=11)
            ax1.text(0.01, 0.74, f"Trade return: {100*float(row['portfolio_return']):.2f}%", fontsize=11)
            ax1.text(0.01, 0.66, f"Headline: {hd['text'][:240]}", fontsize=10, wrap=True)
            ax1.text(0.01, 0.52, f"Class probs [neg, neu, pos] = {np.round(probs, 4)}", fontsize=10)

            _plot_token_bar(ax2, tokens, attrs, title="Top token attributions (positive vs negative influence)")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not regime_tbl.empty:
            fig, ax = plt.subplots(figsize=(11, 6.5))
            x = np.arange(len(regime_tbl))
            ax.bar(x - 0.15, regime_tbl["risk_word_attribution"], width=0.3, label="Risk words", color="#ef4444")
            ax.bar(x + 0.15, regime_tbl["growth_word_attribution"], width=0.3, label="Growth words", color="#22c55e")
            ax.set_xticks(x)
            ax.set_xticklabels(regime_tbl["regime"].astype(str).tolist())
            ax.set_xlabel("HMM regime")
            ax.set_ylabel("Mean absolute attribution")
            ax.set_title("Regime sensitivity: token attribution profile")
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate word-level sentiment explainability case study PDF.")
    ap.add_argument("--config", type=str, default="config/hybrid_config.yaml")
    ap.add_argument("--text-input", type=str, default=None)
    ap.add_argument("--backtest", type=str, default="results/ablation/backtest_Full_Hybrid_Proposed_Momentum_plus_HMM_plus_FinBERT_LoRA.csv")
    ap.add_argument("--out", type=str, default="results/case_study_explainability.pdf")
    ap.add_argument("--ig-steps", type=int, default=24)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    paths = cfg.get("paths", {})
    scfg = cfg.get("sentiment_pipeline", {})

    ecfg = ExplainConfig(
        model_name=str(scfg.get("finbert_model_name", "ProsusAI/finbert")),
        lora_adapter_path=paths.get("finbert_lora_adapter", None),
        text_input=Path(args.text_input or paths.get("text_input", "data/raw/news_text.parquet")),
        backtest_csv=Path(args.backtest),
        nifty_features=Path(paths.get("nifty_features", "data/processed/_NSEI_features.parquet")),
        out_pdf=Path(args.out),
        ig_steps=int(args.ig_steps),
    )

    build_case_study(ecfg)
    print(f"Wrote explainability report: {ecfg.out_pdf}")


if __name__ == "__main__":
    main()
