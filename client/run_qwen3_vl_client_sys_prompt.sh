uv run python ./vllm_serving/client/client_qwen3_vl.py  \
    --model Qwen3-VL-8B-Instruct \
    --image ./vllm_serving/media/image.png \
    --system-prompt ./client/system_prompt/robot_teacher_career_guide.md
