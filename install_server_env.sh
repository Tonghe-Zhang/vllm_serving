#!/usr/bin/env bash
set -euo pipefail

# Minimal environment setup for serving Qwen3-VL-8B-Thinking with vLLM.
# This script only prepares the environment; it does not start the server.

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/4] Creating virtual environment at: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[2/4] Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "[3/4] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[4/4] Installing minimal serving dependencies"
python -m pip install "vllm" "openai>=1.0.0"

cat <<'EOF'

Environment is ready.

Next steps:
  source .venv/bin/activate
  python serve_qwen3_vl.py --attention-backend FLASH_ATTN

Then test:
  curl http://127.0.0.1:8000/v1/models
EOF

