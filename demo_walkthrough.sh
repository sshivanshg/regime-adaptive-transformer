#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD="python3"
if command -v python &>/dev/null; then
    PYTHON_CMD="python"
fi

echo "[1/3] Generating final report and summary assets"
$PYTHON_CMD scripts/generate_final_report.py
$PYTHON_CMD scripts/generate_summary_infographic.py

echo "[2/3] Launching Streamlit dashboard"
streamlit run dashboard/app.py &
DASH_PID=$!

sleep 3

echo "[3/3] Opening FINAL_REPORT.md (if desktop opener available)"
if command -v open >/dev/null 2>&1; then
  open FINAL_REPORT.md || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open FINAL_REPORT.md || true
else
  echo "Open manually: FINAL_REPORT.md"
fi

echo "Dashboard PID: $DASH_PID"
echo "Press Ctrl+C to stop this script and dashboard."
wait $DASH_PID
