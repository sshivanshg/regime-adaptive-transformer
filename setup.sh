#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Turn-key smoke test
python main.py --task smoke-test --config config/hybrid_config.yaml

echo "Setup complete. Next run:"
echo "  python main.py --task all --config config/hybrid_config.yaml --start 2024-01-01 --end 2026-04-16"
