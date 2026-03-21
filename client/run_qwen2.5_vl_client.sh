#!/bin/bash

# First run ssh -N -L 8000:localhost:8000 <server_host> in another terminal to set up port forwarding.
# Then run the client in the current terminal
uv run python ./vllm_serving/client/client_qwen3_vl.py \
    --model SpaceQwen2.5-VL-3B-Instruct \
    --image ./vllm_serving/media/image.jpg
