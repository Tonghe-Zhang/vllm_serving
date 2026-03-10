#!/usr/bin/env bash
set -euo pipefail

# Minimal environment setup for serving Qwen3-VL-8B-Thinking with vLLM.
# This script only prepares the environment; it does not start the server.

uv pip install openai
