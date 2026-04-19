#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

echo "Starting Momentum + Regime dashboard..."
echo "Open: http://localhost:8501"

streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.headless false \
    --theme.base dark \
    --theme.primaryColor "#3b82f6" \
    --theme.backgroundColor "#0f172a" \
    --theme.secondaryBackgroundColor "#1e293b" \
    --theme.textColor "#f1f5f9"

