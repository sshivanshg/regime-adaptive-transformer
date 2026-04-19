"""
NIFTY 500 (current NSE snapshot) × six calendar years (2010–2015): full yfinance pipeline
per year — separate ``data/raw_yf_<year>``, ``data/processed_yf_<year>``, momentum rankings,
and ``results/hmm_vs_flat/yf_<year>/``.

Ranking signal is **pure Ret_21d** via ``momentum_predictions_from_features.py`` (unchanged).

After all runs: print a markdown table, append/replace the documented block in ``RESULTS.md``,
and print NIFTY buy-and-hold per year.

Example::

  . .venv/bin/activate
  python scripts/run_nifty500_annual_hmm_ablation.py \\
    --universe-file scripts/universe/nifty500_nse_survivorship_proxy.txt

  # Recompute metrics only (data already on disk)::
  python scripts/run_nifty500_annual_hmm_ablation.py --skip-download --skip-features --skip-momentum
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
YEARS_DEFAULT = tuple(range(2010, 2016))

RESULTS_MARKER_START = "<!-- NIFTY500_ANNUAL_ABLATION_START -->"
RESULTS_MARKER_END = "<!-- NIFTY500_ANNUAL_ABLATION_END -->"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def nifty_buy_hold_pct(raw_dir: Path, year: int) -> float:
    p = raw_dir / "_NSEI.parquet"
    if not p.is_file():
        return float("nan")
    df = pd.read_parquet(p, columns=["Date", "Adj Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    y = df[df["Date"].dt.year == year].sort_values("Date")
    if len(y) < 2:
        return float("nan")
    a0 = float(y["Adj Close"].iloc[0])
    a1 = float(y["Adj Close"].iloc[-1])
    return (a1 / a0 - 1.0) * 100.0


def read_summary_metrics(out_hmm: Path, bt_start: str, bt_end: str) -> tuple[dict, dict]:
    label = f"{bt_start}_{bt_end}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")[:80]
    summary = out_hmm / slug / "hmm_vs_flat_summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"Missing summary: {summary}")
    df = pd.read_csv(summary)
    hmm = df[df["variant"] == "hmm_conditioned_portfolio"].iloc[0].to_dict()
    flat = df[df["variant"] == "regime_agnostic_flat_sizing"].iloc[0].to_dict()
    return hmm, flat


def equity_ok_count(raw_dir: Path) -> int:
    stats = raw_dir / "_fetch_stats.json"
    if not stats.is_file():
        return -1
    data = json.loads(stats.read_text(encoding="utf-8"))
    return int(data.get("equities_downloaded_ok", -1))


def fmt_pct(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{nd}f}%"


def fmt_num(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{nd}f}"


def patch_results_md(results_path: Path, block: str) -> None:
    text = results_path.read_text(encoding="utf-8")
    wrapped = f"{RESULTS_MARKER_START}\n{block.rstrip()}\n{RESULTS_MARKER_END}"
    if RESULTS_MARKER_START in text and RESULTS_MARKER_END in text:
        pattern = re.compile(
            re.escape(RESULTS_MARKER_START) + r".*?" + re.escape(RESULTS_MARKER_END),
            re.DOTALL,
        )
        new_text = pattern.sub(wrapped, text, count=1)
    else:
        needle = "**Artifacts (per historical window):**"
        if needle not in text:
            raise SystemExit(f"Cannot find {needle!r} in {results_path}")
        new_text = text.replace(needle, wrapped + "\n\n" + needle, 1)
    results_path.write_text(new_text, encoding="utf-8")
    print(f"\nUpdated {results_path} ({RESULTS_MARKER_START} … {RESULTS_MARKER_END})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="NIFTY 500 annual HMM ablation 2010–2015")
    ap.add_argument(
        "--universe-file",
        type=Path,
        default=ROOT / "scripts" / "universe" / "nifty500_nse_survivorship_proxy.txt",
        help="One SYMBOL.NS per line (default: NIFTY 500 proxy from repo)",
    )
    ap.add_argument("--years", type=int, nargs="*", default=list(YEARS_DEFAULT))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-momentum", action="store_true")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="Do not run pipelines; only aggregate existing raw/summary files into the table",
    )
    ap.add_argument("--no-write-results-md", action="store_true")
    args = ap.parse_args()

    uni = Path(args.universe_file).expanduser().resolve()
    if not uni.is_file():
        raise SystemExit(f"Universe file not found: {uni}")

    py = sys.executable
    rows: list[dict] = []

    for year in sorted(set(args.years)):
        tag = str(year)
        start = f"{year}-01-01"
        end_ex = f"{year + 1}-01-01"
        bt_end = f"{year}-12-31"
        raw_dir = ROOT / "data" / f"raw_yf_{tag}"
        out_hmm = ROOT / "results" / "hmm_vs_flat" / f"yf_{tag}"

        if not args.report_only:
            cmd = [
                py,
                str(ROOT / "scripts" / "run_yf_hmm_ablation.py"),
                "--start",
                start,
                "--end-exclusive",
                end_ex,
                "--bt-start",
                start,
                "--bt-end",
                bt_end,
                "--tag",
                tag,
                "--universe-file",
                str(uni),
            ]
            if args.skip_download:
                cmd.append("--skip-download")
            if args.skip_features:
                cmd.append("--skip-features")
            if args.skip_momentum:
                cmd.append("--skip-momentum")
            run(cmd)

        n_eq = equity_ok_count(raw_dir)
        nifty_bh = nifty_buy_hold_pct(raw_dir, year)
        hmm, flat = read_summary_metrics(out_hmm, start, bt_end)

        rows.append(
            {
                "year": year,
                "n_eq": n_eq,
                "hmm_sharpe": float(hmm["sharpe"]),
                "hmm_cagr": float(hmm["cagr_pct"]),
                "hmm_mdd": float(hmm["max_dd_pct"]),
                "flat_sharpe": float(flat["sharpe"]),
                "flat_cagr": float(flat["cagr_pct"]),
                "flat_mdd": float(flat["max_dd_pct"]),
                "nifty_bh_pct": nifty_bh,
            }
        )

    lines = [
        "### NIFTY 500 annual sub-windows (2010–2015, momentum `Ret_21d` only)",
        "",
        "Universe: **current NIFTY 500** snapshot (`scripts/universe/nifty500_nse_survivorship_proxy.txt`). "
        "Separate `raw_yf_<year>` / `processed_yf_<year>` / `hmm_vs_flat/yf_<year>/` per calendar year.",
        "",
        "| Year | Equities OK (requested 500) | HMM Sharpe | HMM CAGR | HMM Max DD | Flat Sharpe | Flat CAGR | Flat Max DD | NIFTY B&H (year) |",
        "|------|----------------------------|------------|----------|------------|-------------|-----------|-------------|------------------|",
    ]
    for r in rows:
        req = 500
        ne = r["n_eq"] if r["n_eq"] >= 0 else "n/a"
        lines.append(
            f"| {r['year']} | **{ne}** ({req}) | {fmt_num(r['hmm_sharpe'])} | {fmt_pct(r['hmm_cagr'])} | "
            f"{fmt_pct(r['hmm_mdd'])} | {fmt_num(r['flat_sharpe'])} | {fmt_pct(r['flat_cagr'])} | "
            f"{fmt_pct(r['flat_mdd'])} | {fmt_pct(r['nifty_bh_pct'])} |"
        )
    lines += [
        "",
        "**Caveats (this block):**",
        "",
        "1. **Membership:** NIFTY 500 **current** list used as a survivorship-biased proxy; true 2010–2015 "
        "constituents are unavailable. This likely **inflates** backtested returns versus a point-in-time index.",
        "2. **Single-year Sharpe:** ~12 monthly rebalance observations per year — **noisy**; use as a coarse screen only.",
        "3. **CAGR:** Single-calendar-year CAGR is **not** directly comparable to multi-year CAGR in the four-window table above.",
        "",
        "**Reproduce:** `python scripts/run_nifty500_annual_hmm_ablation.py --universe-file scripts/universe/nifty500_nse_survivorship_proxy.txt`",
        "",
    ]
    block = "\n".join(lines)
    print("\n" + "=" * 72)
    print(block)
    print("=" * 72 + "\n")

    if not args.no_write_results_md:
        patch_results_md(ROOT / "RESULTS.md", block)


if __name__ == "__main__":
    main()
