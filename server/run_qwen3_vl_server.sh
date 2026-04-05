#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="$SCRIPT_DIR/.."
cd "$REPO_DIR"

python server/serve_vlm.py \
--model-type qwen3-vl-8b \
--model-base-dir /usr0/PretrainedModels \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.65