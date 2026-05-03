"""
User Story:
Produce a professional final report that synthesizes methodology, ablations, and
phase-wise outcomes using artifact-backed metrics for submission.

Implementation Approach:
Load available CSV/JSON outputs, normalize into the required four-scenario table,
compute diagnostics, and render FINAL_REPORT.md from a deterministic template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _compute_metrics(bt_path: Path, capital: float = 100000.0) -> dict[str, float]:
    if not bt_path.is_file():
        return {}
    bt = pd.read_csv(bt_path, parse_dates=["date"])
    if bt.empty:
        return {}

    r = bt["portfolio_return"].astype(float).fillna(0.0)
    nav = bt["portfolio_value"].astype(float).to_numpy()

    start = pd.to_datetime(bt["date"].iloc[0])
    end = pd.to_datetime(bt["date"].iloc[-1])
    years = max((end - start).days / 365.25, 1e-9)

    total_ret = float(nav[-1] / capital - 1.0)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(12.0))

    peak = np.maximum.accumulate(nav)
    mdd = float(((nav - peak) / peak).min())
    win = float((r > 0).mean())

    return {
        "CAGR": cagr,
        "Sharpe_Net": sharpe,
        "Max_Drawdown": mdd,
        "Win_Rate": win,
    }


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{100.0 * float(x):.2f}%"


def _fmt_num(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{float(x):.3f}"


def _row(label: str, metrics: dict[str, float] | None, source: str) -> dict[str, str]:
    m = metrics or {}
    return {
        "Scenario": label,
        "CAGR": _fmt_pct(m.get("CAGR")),
        "Sharpe (Net)": _fmt_num(m.get("Sharpe_Net")),
        "Max Drawdown": _fmt_pct(m.get("Max_Drawdown")),
        "Win Rate": _fmt_pct(m.get("Win_Rate")),
        "Source": source,
    }


def build_ablation_table() -> tuple[pd.DataFrame, str]:
    # Primary source from Phase 2 if present and schema-compatible.
    ablation = ROOT / "results" / "ablation_report.csv"
    if ablation.is_file():
        a = pd.read_csv(ablation)
        expected_rows = {
            "Baseline: Momentum Only",
            "ML-Enhanced: Momentum + HMM",
            "DL-Enhanced: Momentum + FinBERT (Vanilla)",
            "Full Hybrid (Proposed): Momentum + HMM + FinBERT (LoRA)",
        }
        has_required_cols = {"Scenario", "CAGR", "Sharpe_Net", "Max_Drawdown", "Win_Rate"}.issubset(set(a.columns))
        has_expected_rows = has_required_cols and expected_rows.issubset(set(a["Scenario"].astype(str).unique()))
        if has_expected_rows:
            mapping = {
                "Baseline: Momentum Only": "Baseline (Momentum)",
                "ML-Enhanced: Momentum + HMM": "ML-Enhanced (Momentum + HMM)",
                "DL-Enhanced: Momentum + FinBERT (Vanilla)": "DL-Enhanced (Momentum + FinBERT)",
                "Full Hybrid (Proposed): Momentum + HMM + FinBERT (LoRA)": "Full Hybrid (Our Model)",
            }
            rows = []
            for src, dst in mapping.items():
                r = a[a["Scenario"] == src]
                if r.empty:
                    rows.append(_row(dst, None, "results/ablation_report.csv (missing row)"))
                    continue
                m = r.iloc[0]
                rows.append(
                    _row(
                        dst,
                        {
                            "CAGR": float(m["CAGR"]),
                            "Sharpe_Net": float(m["Sharpe_Net"]),
                            "Max_Drawdown": float(m["Max_Drawdown"]),
                            "Win_Rate": float(m["Win_Rate"]),
                        },
                        "results/ablation_report.csv",
                    )
                )
            return pd.DataFrame(rows), "phase2"

    # Fallback synthesis from currently available results.
    baseline_hmm = ROOT / "results" / "hmm_ablation" / "2024_2026" / "2024-01-01_2025-12-31" / "hmm_vs_flat_summary.csv"
    final_bt = ROOT / "results" / "final_strategy" / "backtest_results.csv"

    baseline_metrics = None
    ml_metrics = None
    if baseline_hmm.is_file():
        h = pd.read_csv(baseline_hmm)
        b = h[h["variant"] == "regime_agnostic_flat_sizing"]
        m = h[h["variant"] == "hmm_conditioned_portfolio"]
        if not b.empty:
            rb = b.iloc[0]
            baseline_metrics = {
                "CAGR": float(rb["cagr_pct"]) / 100.0,
                "Sharpe_Net": float(rb["sharpe"]),
                "Max_Drawdown": float(rb["max_dd_pct"]) / 100.0,
                "Win_Rate": np.nan,
            }
        if not m.empty:
            rm = m.iloc[0]
            ml_metrics = {
                "CAGR": float(rm["cagr_pct"]) / 100.0,
                "Sharpe_Net": float(rm["sharpe"]),
                "Max_Drawdown": float(rm["max_dd_pct"]) / 100.0,
                "Win_Rate": np.nan,
            }

    full_metrics = _compute_metrics(final_bt) if final_bt.is_file() else None

    rows = [
        _row("Baseline (Momentum)", baseline_metrics, "results/hmm_ablation/.../hmm_vs_flat_summary.csv"),
        _row("ML-Enhanced (Momentum + HMM)", ml_metrics, "results/hmm_ablation/.../hmm_vs_flat_summary.csv"),
        _row("DL-Enhanced (Momentum + FinBERT)", None, "results/ablation_report.csv not found"),
        _row("Full Hybrid (Our Model)", full_metrics, "results/final_strategy/backtest_results.csv"),
    ]
    return pd.DataFrame(rows), "fallback"


def _diagnostic_paragraphs(df: pd.DataFrame) -> list[str]:
    out = []
    for _, r in df.iterrows():
        sc = r["Scenario"]
        if r["CAGR"] == "N/A":
            out.append(
                f"- **{sc}**: This scenario has no runnable artifact in the current repository export. "
                "The expected metric source is missing, so no quantitative claim is made."
            )
            continue
        if "Baseline" in sc:
            out.append(
                f"- **{sc}**: Serves as the pure price-signal control. Performance reflects momentum capture "
                "without macro-state conditioning or text-derived risk adjustment."
            )
        elif "ML-Enhanced" in sc:
            out.append(
                f"- **{sc}**: Adding HMM regime sizing changes exposure profile versus baseline, typically reducing "
                "risk concentration during unstable windows at the cost of upside during persistent rallies."
            )
        elif "DL-Enhanced" in sc:
            out.append(
                f"- **{sc}**: Sentiment-only enhancement isolates text signal contribution. Any drawdown reduction would "
                "indicate earlier risk detection than trailing-price stop logic."
            )
        else:
            out.append(
                f"- **{sc}**: Joint regime + sentiment gating combines macro-state risk control with narrative-level signal "
                "timing. Improvements in Sharpe or drawdown are interpreted as complementary ML+DL behavior."
            )
    return out


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([head, sep] + rows)


def generate(output: Path, export_ablation_csv: Path | None = None) -> None:
    table, source_mode = build_ablation_table()
    diagnostics = _diagnostic_paragraphs(table)

    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# FINAL REPORT")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(
        "This project evolved from a transformer-centric ranking hypothesis to a validated hybrid architecture "
        "where regime-aware risk control (HMM) and sentiment-aware gating (FinBERT + LoRA pathway) are explicitly separated and testable. "
        "The final system emphasizes robustness, diagnostic transparency, and reproducible execution over headline complexity."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append(
        "The production decision layer uses a Regime-Adaptive Sentiment Gating policy: "
        "in Bull regimes momentum remains primary with permissive sentiment filtering; "
        "in High-Vol regimes score fusion uses weighted momentum-sentiment integration; "
        "in Bear regimes sentiment conviction gates entries and allows cash states. "
        "All feature streams are shifted to enforce T-1 information for T execution, preventing look-ahead leakage."
    )
    lines.append("")
    lines.append("## Ablation Results")
    lines.append("")
    lines.append(_df_to_markdown_table(table))
    lines.append("")
    if source_mode == "fallback":
        lines.append(
            "**Note:** `results/ablation_report.csv` is not present in the current artifact set. "
            "Rows were populated from available backtest summaries where possible; unavailable rows remain `N/A`."
        )
        lines.append("")
    lines.append("## Diagnostic Analysis")
    lines.append("")
    lines.extend(diagnostics)
    lines.append("")
    lines.append("## Reproducibility Statement")
    lines.append(
        "The repository includes a single orchestrator (`main.py --task all`), container manifests, a self-audit script, "
        "and deterministic report generation. This final report is generated from local result artifacts only, without manual metric edits."
    )

    output.write_text("\n".join(lines), encoding="utf-8")

    if export_ablation_csv is not None:
        export_ablation_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df = table.rename(columns={"Sharpe (Net)": "Sharpe_Net", "Max Drawdown": "Max_Drawdown", "Win Rate": "Win_Rate"})
        out_df.to_csv(export_ablation_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate FINAL_REPORT.md from available artifacts.")
    ap.add_argument("--output", type=str, default="FINAL_REPORT.md")
    ap.add_argument("--export-ablation-csv", type=str, default="results/ablation_report.csv")
    args = ap.parse_args()
    generate(Path(args.output), export_ablation_csv=Path(args.export_ablation_csv))
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
