#!/usr/bin/env python3
"""Unified client for any VLM served by vLLM (OpenAI-compatible API).

Supports:
  - One-shot inference (--prompt) or interactive chat
  - Image attachment (image placed BEFORE text, required by Gemma 4)
  - Optional system prompt from a file
  - Streaming responses
  - Thinking mode control (--no-thinking)
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with any VLM via vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True, help="Served model name (from /v1/models).")
    parser.add_argument("--image", type=Path, default=None, help="Local image to attach.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="One-shot prompt (non-interactive). Omit for interactive chat.")
    parser.add_argument("--system-prompt", type=Path, default=None,
                        help="Path to .md/.txt system prompt file.")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Suppress <think> blocks (Qwen thinking models).")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def build_content(text: str, image_path: Path | None) -> str | list:
    """Build message content. Image comes FIRST (required by Gemma 4, fine for others)."""
    if image_path is None or not image_path.exists():
        return text
    data = base64.b64encode(image_path.read_bytes()).decode()
    suffix = image_path.suffix.lstrip(".").replace("jpg", "jpeg")
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{data}"}},
        {"type": "text", "text": text},
    ]


RED = "\033[91m"
RESET = "\033[0m"


def warn(msg: str) -> None:
    """Print a red warning to stderr."""
    print(f"{RED}Warning: {msg}{RESET}", file=sys.stderr)


def stream_response(client, model, messages, max_tokens, temperature, top_p, extra) -> str | None:
    """Stream a chat completion, returning the full reply or None on error."""
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, stream=True,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, **extra,
        )
    except Exception as e:
        warn(f"{type(e).__name__}: {e}")
        return None
    reply = ""
    try:
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
    except Exception as e:
        warn(f"Stream interrupted: {e}")
        return reply or None
    print()
    return reply


def one_shot(client, model, prompt, image, system_prompt, max_tokens, temperature, top_p, extra):
    """Send a single prompt, print the response, then continue interactively."""
    history = []
    if system_prompt and system_prompt.exists():
        history.append({"role": "system", "content": system_prompt.read_text().strip()})
    history.append({"role": "user", "content": build_content(prompt, image)})

    print(f"Model : {model}")
    if image:
        print(f"Image : {image}")
    print(f"Prompt: {prompt}\n")
    print("Response:", flush=True)

    reply = stream_response(client, model, history, max_tokens, temperature, top_p, extra)
    if reply is not None:
        history.append({"role": "assistant", "content": reply})
    print()

    # Continue into interactive mode
    interactive(client, model, None, None, max_tokens, temperature, top_p, extra,
                history=history)
    return reply


def interactive(client, model, image, system_prompt, max_tokens, temperature, top_p, extra,
                *, history: list[dict] | None = None):
    """Multi-turn interactive chat."""
    if history is None:
        history = []
        if system_prompt and system_prompt.exists():
            sys_text = system_prompt.read_text().strip()
            history.append({"role": "system", "content": sys_text})
            print(f"System prompt: {system_prompt.name}")

    print(f"Chatting with {model} — type 'quit' or Ctrl-C to exit.\n")

    first_turn = not any(m["role"] == "user" for m in history)
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Bye.")
            break
        if not user_input:
            continue

        img = image if first_turn else None
        content = build_content(user_input, img)
        history.append({"role": "user", "content": content})
        first_turn = False

        print("\nAssistant: ", end="", flush=True)
        reply = stream_response(client, model, history, max_tokens, temperature, top_p, extra)
        if reply is not None:
            history.append({"role": "assistant", "content": reply})
        print()


def main():
    args = parse_args()
    if args.image is not None and not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else {}

    if args.prompt:
        one_shot(client, args.model, args.prompt, args.image, args.system_prompt,
                 args.max_tokens, args.temperature, args.top_p, extra)
    else:
        interactive(client, args.model, args.image, args.system_prompt,
                    args.max_tokens, args.temperature, args.top_p, extra)


if __name__ == "__main__":
    main()
