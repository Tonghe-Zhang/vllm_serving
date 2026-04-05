#!/bin/bash

# First run ssh -N -L 8000:localhost:8000 <server_host> in another terminal to set up port forwarding.
# Then run the client in the current terminal
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
cd "$REPO_DIR"

python client/client_vlm.py \
    --model SpaceQwen2.5-VL-3B-Instruct \
    --image media/image.jpg
