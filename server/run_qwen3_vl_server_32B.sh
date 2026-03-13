#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../vllm_serving"

uv run python server/serve_qwen3_vl.py \
--model-type instruct-32b \
--model-base-dir ../PretrainedModels \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.65