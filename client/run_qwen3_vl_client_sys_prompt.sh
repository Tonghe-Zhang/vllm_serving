# First run ssh -N -L 8000:localhost:8000 LeCAR_4xRTX6000BlackWell_97GB in another terminal to set up port forwarding.

uv run python ./vllm_serving/client/client_qwen3_vl.py  \
    --model Qwen3-VL-8B-Instruct \
    --image ./vllm_serving/media/image-box.png \
    --system-prompt ./client/system_prompt/robot_teacher_career_guide.md
