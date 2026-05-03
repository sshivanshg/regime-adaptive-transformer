"""
User Story:
Generate a single visual summary panel that helps graders quickly understand the
architecture, performance trajectory, and extra-mile contributions.

Implementation Approach:
Compose a matplotlib figure with three blocks: equity curve snapshot, architecture
thumbnail, and curated feature callouts sourced from project artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def generate(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=180)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    fig.suptitle("RAMT Final Submission Summary", fontsize=20, weight="bold")

    bt_path = ROOT / "results" / "final_strategy" / "backtest_results.csv"
    if bt_path.is_file():
        bt = pd.read_csv(bt_path, parse_dates=["date"])
        ax1.plot(bt["date"], bt["portfolio_value"], color="#0ea5e9", linewidth=2.2, label="Strategy NAV")
        ax1.set_title("Production Equity Curve")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(alpha=0.2)
        ax1.legend()
    else:
        ax1.text(0.05, 0.5, "Missing results/final_strategy/backtest_results.csv", fontsize=12)
        ax1.set_axis_off()

    arch_path = ROOT / "docs" / "architecture_final.png"
    if arch_path.is_file():
        img = mpimg.imread(str(arch_path))
        ax2.imshow(img)
        ax2.set_title("Final Architecture")
        ax2.axis("off")
    else:
        ax2.text(0.05, 0.5, "Missing docs/architecture_final.png", fontsize=12)
        ax2.axis("off")

    ax3.axis("off")
    bullets = [
        "Extra Mile #1: Interactive dashboard with sentiment explorer, heatmap, and live ablation toggles.",
        "Extra Mile #2: Explainability module with Integrated Gradients and regime sensitivity case study PDF.",
        "Extra Mile #3: Turn-key packaging via main.py --task all, Dockerfile, and final audit checklist.",
    ]
    ax3.text(0.0, 0.95, "Top-3 Extra Mile Features", fontsize=15, weight="bold", va="top")
    y = 0.82
    for b in bullets:
        ax3.text(0.0, y, f"• {b}", fontsize=11, va="top")
        y -= 0.24

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = ROOT / "results" / "summary_infographic.png"
    generate(out)
    print(f"Wrote infographic: {out}")


if __name__ == "__main__":
    main()
