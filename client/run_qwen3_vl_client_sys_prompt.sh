cd /usr0/tonghez/vllm_serving
uv run python client/client_qwen3_vl.py  \
--model Qwen3-VL-8B-Instruct \
--image ./media/image.png \
--system-prompt ./client/system_prompt/robot_teacher_career_guide.md
