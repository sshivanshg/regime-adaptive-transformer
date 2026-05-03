#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate
python main.py --task all --config config/hybrid_config.yaml

echo "Run complete. Launch dashboard with:"
echo "  streamlit run dashboard/app.py"
