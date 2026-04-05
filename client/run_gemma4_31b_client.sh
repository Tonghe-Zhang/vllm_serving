#!/usr/bin/env bash
# One-shot inference with Gemma-4-31B: identify objects on blue plate / in brown box.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
cd "$REPO_DIR"

python client/client_vlm.py \
    --model gemma-4-31B-it \
    --base-url http://localhost:8001/v1 \
    --image media/image.png \
    --prompt "List all the objects you see in this image. Which objects are on the blue plate and which objects are in the brown box?"
