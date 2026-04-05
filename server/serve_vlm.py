#!/usr/bin/env python3
"""Launch a minimal OpenAI-compatible vLLM server for any supported VLM.

Supported model families:
  - Qwen3-VL (8B thinking/instruct, 32B instruct)
  - SpaceQwen2.5-VL-3B
  - Gemma-4-31B

Add new models by appending to KNOWN_MODELS below.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


KNOWN_MODELS = {
    # Qwen family
    "qwen3-vl-8b-thinking":  "Qwen3-VL-8B-Thinking",
    "qwen3-vl-8b":           "Qwen3-VL-8B-Instruct",
    "qwen3-vl-32b":          "Qwen3-VL-32B-Instruct",
    "spacequen2.5-vl-3b":    "SpaceQwen2.5-VL-3B-Instruct",
    # Gemma family
    "gemma4-31b":             "gemma-4-31B-it",
}

# Shortcuts mapping old --model-type names → canonical keys (backward compat)
_ALIASES = {
    "thinking":    "qwen3-vl-8b-thinking",
    "instruct":    "qwen3-vl-8b",
    "instruct-32b":"qwen3-vl-32b",
    "spatial":     "spacequen2.5-vl-3b",
}


def _resolve_model_key(raw: str) -> str:
    """Accept either canonical key or legacy alias."""
    if raw in KNOWN_MODELS:
        return raw
    if raw in _ALIASES:
        return _ALIASES[raw]
    raise argparse.ArgumentTypeError(
        f"Unknown model type '{raw}'. "
        f"Choices: {', '.join(sorted(set(list(KNOWN_MODELS) + list(_ALIASES))))}"
    )


def parse_args() -> argparse.Namespace:
    all_choices = sorted(set(list(KNOWN_MODELS) + list(_ALIASES)))
    parser = argparse.ArgumentParser(
        description="Serve any VLM with vLLM (OpenAI-compatible API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Known model types:\n  " + "\n  ".join(all_choices),
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", type=Path, default=None,
        help="Explicit path to a local model directory.",
    )
    model_group.add_argument(
        "--model-type", type=_resolve_model_key, default=None,
        help=f"Shorthand model variant. Choices: {', '.join(all_choices)}",
    )

    parser.add_argument("--model-base-dir", type=Path, default=None,
        help="Base directory containing downloaded models. "
             "Falls back to MODEL_BASE_DIR env var.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None,
        help="Name returned by /v1/models. Defaults to model folder name.")
    parser.add_argument("--dtype", default="auto",
        choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--limit-images-per-prompt", type=int, default=4)
    parser.add_argument("--attention-backend", default=None,
        help="Attention backend (FLASH_ATTN, FLASHINFER, XFORMERS). "
             "Default: let vLLM auto-select.")
    parser.add_argument("--chat-template", default=None,
        help="Path to a Jinja2 chat template file.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--disable-generation-config", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model is not None:
        return args.model.expanduser().resolve()

    if args.model_type is not None:
        base_dir = args.model_base_dir or (
            Path(os.environ["MODEL_BASE_DIR"])
            if os.environ.get("MODEL_BASE_DIR") else None
        )
        if base_dir is None:
            print("Error: --model-type requires --model-base-dir or MODEL_BASE_DIR env.",
                  file=sys.stderr)
            sys.exit(1)
        folder = KNOWN_MODELS[args.model_type]
        return base_dir.expanduser().resolve() / folder

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
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", served_name,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
        "--limit-mm-per-prompt", f'{{"image": {args.limit_images_per_prompt}}}',
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    if args.attention_backend:
        cmd.extend(["--attention-backend", args.attention_backend])

    if args.chat_template:
        cmd.extend(["--chat-template", args.chat_template])

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
