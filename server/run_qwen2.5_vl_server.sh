#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."

cd "$REPO_DIR"

python server/serve_vlm.py \
--model /usr0/PretrainedModels/SpaceQwen2.5-VL-3B-Instruct \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.65
