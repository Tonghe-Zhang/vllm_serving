SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
cd "$REPO_DIR"

python client/client_vlm.py \
    --model Qwen3-VL-32B-Instruct \
    --image media/image.png
