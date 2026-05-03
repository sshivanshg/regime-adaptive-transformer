"""
User Story:
Produce a publication-ready architecture diagram that clearly communicates the
Regime-Adaptive Sentiment Gating pipeline for reports and README artifacts.

Implementation Approach:
Render a high-resolution static PNG with matplotlib patches and arrows so the
diagram is reproducible in any Python environment without external GUI tools.
The figure annotates tensor shapes, ML vs DL components, and LoRA notation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PALETTE = {
    "bg": "#f8fafc",
    "ml": "#0ea5e9",
    "dl": "#f97316",
    "fusion": "#22c55e",
    "text": "#0f172a",
    "muted": "#475569",
}


def _box(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> None:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=color,
        facecolor="#ffffff",
        alpha=0.98,
    )
    ax.add_patch(p)
    ax.text(x + 0.02, y + h - 0.05, title, fontsize=10.5, fontweight="bold", color=PALETTE["text"])
    ax.text(x + 0.02, y + h - 0.11, subtitle, fontsize=9, color=PALETTE["muted"])


def _arrow(ax, a: tuple[float, float], b: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def generate(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "Regime-Adaptive Transformer: Final Hybrid Architecture",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["text"],
    )
    ax.text(
        0.02,
        0.92,
        "ML lane (blue): momentum + HMM | DL lane (orange): FinBERT + LoRA | Fusion lane (green): regime-adaptive gating",
        fontsize=10,
        color=PALETTE["muted"],
    )

    # Inputs
    _box(ax, 0.03, 0.74, 0.2, 0.14, "Price Features", "X_price: [B, T, F=10]", PALETTE["ml"])
    _box(ax, 0.03, 0.54, 0.2, 0.14, "News/Headlines", "X_text: [B, N_tokens]", PALETTE["dl"])
    _box(ax, 0.03, 0.34, 0.2, 0.14, "NIFTY Regime Stream", "r_t from HMM: [B, 1]", PALETTE["ml"])

    # ML lane
    _box(ax, 0.30, 0.74, 0.22, 0.14, "Momentum Encoder", "Ret_21d ranking -> m_t: [B, 1]", PALETTE["ml"])
    _box(ax, 0.56, 0.74, 0.2, 0.14, "HMM Regime Classifier", "P(r_t|x): {Bull, HighVol, Bear}", PALETTE["ml"])

    # DL lane
    _box(ax, 0.30, 0.54, 0.22, 0.14, "FinBERT Encoder", "h_text: [B, N, H]", PALETTE["dl"])
    _box(ax, 0.56, 0.54, 0.2, 0.14, "LoRA Adapter Block", r"$W = W_0 + \Delta W = W_0 + BA$", PALETTE["dl"])
    _box(ax, 0.79, 0.54, 0.17, 0.14, "Sentiment Head", "s_t in [-1,1], c_t in [0,1]", PALETTE["dl"])

    # Fusion lane
    _box(ax, 0.30, 0.30, 0.28, 0.16, "Regime-Adaptive Sentiment Gating", "Bull: momentum + soft sentiment filter\nHighVol: 0.6*m + 0.4*s\nBear: sentiment-conviction gate", PALETTE["fusion"])
    _box(ax, 0.63, 0.30, 0.18, 0.16, "Portfolio Constructor", "Top-K selection + sizing", PALETTE["fusion"])
    _box(ax, 0.84, 0.30, 0.13, 0.16, "Backtest Engine", "NAV_t, Sharpe, MDD", PALETTE["fusion"])

    # Explainability
    _box(ax, 0.56, 0.08, 0.25, 0.14, "Explainability", "Integrated Gradients + Regime sensitivity", "#8b5cf6")

    # Arrows
    _arrow(ax, (0.23, 0.81), (0.30, 0.81), PALETTE["ml"])
    _arrow(ax, (0.52, 0.81), (0.56, 0.81), PALETTE["ml"])

    _arrow(ax, (0.23, 0.61), (0.30, 0.61), PALETTE["dl"])
    _arrow(ax, (0.52, 0.61), (0.56, 0.61), PALETTE["dl"])
    _arrow(ax, (0.76, 0.61), (0.79, 0.61), PALETTE["dl"])

    _arrow(ax, (0.23, 0.41), (0.30, 0.38), PALETTE["ml"])
    _arrow(ax, (0.66, 0.74), (0.46, 0.46), PALETTE["fusion"])
    _arrow(ax, (0.875, 0.54), (0.50, 0.46), PALETTE["fusion"])
    _arrow(ax, (0.44, 0.74), (0.40, 0.46), PALETTE["fusion"])

    _arrow(ax, (0.58, 0.38), (0.63, 0.38), PALETTE["fusion"])
    _arrow(ax, (0.81, 0.38), (0.84, 0.38), PALETTE["fusion"])
    _arrow(ax, (0.90, 0.30), (0.75, 0.22), "#8b5cf6")

    ax.text(0.03, 0.03, "Color code: ML=blue, DL=orange, Fusion=green. Shapes are shown as [Batch, Seq, Hidden/Feature].", fontsize=9, color=PALETTE["muted"])

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = Path("docs/architecture_final.png")
    generate(out)
    print(f"Wrote diagram: {out}")


if __name__ == "__main__":
    main()
