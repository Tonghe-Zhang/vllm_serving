#!/usr/bin/env python3
"""Launch a minimal OpenAI-compatible vLLM server for Qwen3-VL models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


KNOWN_MODELS = {
    "thinking": "Qwen3-VL-8B-Thinking",
    "instruct": "Qwen3-VL-8B-Instruct",
    "instruct-32b": "Qwen3-VL-32B-Instruct",
    "spatial": "SpaceQwen2.5-VL-3B-Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a Qwen3-VL model with vLLM (OpenAI-compatible API)."
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Explicit path to a local model directory.",
    )
    model_group.add_argument(
        "--model-type",
        choices=list(KNOWN_MODELS.keys()),
        default=None,
        help=(
            "Shorthand model variant. The script will look for the model under "
            "MODEL_BASE_DIR/<model-folder> (set MODEL_BASE_DIR env var or use "
            "--model-base-dir). Choices: " + ", ".join(KNOWN_MODELS.keys())
        ),
    )

    parser.add_argument(
        "--model-base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory that contains downloaded model folders. "
            "Used together with --model-type. "
            "Falls back to the MODEL_BASE_DIR environment variable if not set."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the HTTP server.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the HTTP server.")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model name returned by /v1/models. Defaults to the model folder name.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Weights/data type for inference.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum context length to allocate.",
    )
    parser.add_argument(
        "--limit-images-per-prompt",
        type=int,
        default=4,
        help="Maximum number of images per request (multimodal guardrail).",
    )
    parser.add_argument(
        "--attention-backend",
        default="FLASH_ATTN",
        help="Attention backend passed to vLLM (e.g. FLASH_ATTN, XFORMERS).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key check. Keep None for local testing.",
    )
    parser.add_argument(
        "--disable-generation-config",
        action="store_true",
        help="Use vLLM defaults instead of the model's generation_config.json.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM may use per device.",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    """Return an absolute model path from --model or --model-type + base dir."""
    import os

    if args.model is not None:
        return args.model.expanduser().resolve()

    if args.model_type is not None:
        base_dir = args.model_base_dir
        if base_dir is None:
            env_base = os.environ.get("MODEL_BASE_DIR")
            if env_base:
                base_dir = Path(env_base)
            else:
                print(
                    "Error: --model-type requires either --model-base-dir or the "
                    "MODEL_BASE_DIR environment variable to be set.",
                    file=sys.stderr,
                )
                sys.exit(1)
        folder = KNOWN_MODELS[args.model_type]
        return base_dir.expanduser().resolve() / folder

    # Neither flag given — prompt interactively
    raw = input("Enter path to model directory: ").strip()
    if not raw:
        print("No model path provided.", file=sys.stderr)
        sys.exit(1)
    return Path(raw).expanduser().resolve()


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args)

    if not model_path.exists():
        print(f"Model path does not exist: {model_path}", file=sys.stderr)
        return 1

    served_name = args.served_model_name or model_path.name

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", served_name,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
        "--attention-backend", args.attention_backend,
        "--limit-mm-per-prompt", f'{{"image": {args.limit_images_per_prompt}}}',    
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])

    if args.disable_generation_config:
        cmd.extend(["--generation-config", "vllm"])

    print(f"Model   : {model_path}")
    print(f"Serving as: {served_name}")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print("Command :", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())