#!/usr/bin/env bash
# Serve Gemma-4-31B VLM on GPUs 0,1 (most free memory).
# vLLM auto-selects the best attention backend (FlashInfer by default in v0.19+).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES=1

python server/serve_vlm.py \
    --model-type gemma4-31b \
    --model-base-dir /usr0/PretrainedModels \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --limit-images-per-prompt 4 \
    --chat-template /usr0/PretrainedModels/gemma-4-31B-it/chat_template.jinja
