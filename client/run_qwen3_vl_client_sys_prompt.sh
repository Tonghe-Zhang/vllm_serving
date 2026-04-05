# First run ssh -N -L 8000:localhost:8000 LeCAR_4xRTX6000BlackWell_97GB in another terminal to set up port forwarding.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
cd "$REPO_DIR"

python client/client_vlm.py \
    --model Qwen3-VL-8B-Instruct \
    --image media/image-box.png \
    --system-prompt client/system_prompt/robot_teacher_career_guide.md
