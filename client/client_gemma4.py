#!/usr/bin/env python3
"""Client for Gemma-4-31B VLM served by vLLM.

Supports both interactive chat and one-shot inference via --prompt.
Image content is placed BEFORE text (required by Gemma 4).
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Gemma-4-31B VLM.")
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="gemma-4-31B")
    parser.add_argument("--image", type=Path, default=None, help="Local image to attach.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="One-shot prompt (non-interactive). If omitted, enters interactive mode.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def build_content(text: str, image_path: Path | None) -> str | list:
    """Build message content. Image comes FIRST for Gemma 4."""
    if image_path is None or not image_path.exists():
        return text
    data = base64.b64encode(image_path.read_bytes()).decode()
    suffix = image_path.suffix.lstrip(".").replace("jpg", "jpeg")
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{data}"}},
        {"type": "text", "text": text},
    ]


def one_shot(client: OpenAI, model: str, prompt: str, image: Path | None,
             max_tokens: int, temperature: float, top_p: float):
    """Send a single prompt and print the response."""
    content = build_content(prompt, image)
    messages = [{"role": "user", "content": content}]

    print(f"Prompt: {prompt}")
    if image:
        print(f"Image : {image}")
    print(f"Model : {model}\n")
    print("Response:", flush=True)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    reply = ""
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        reply += delta
    print("\n")
    return reply


def interactive(client: OpenAI, model: str, image: Path | None,
                max_tokens: int, temperature: float, top_p: float):
    """Interactive multi-turn chat."""
    history: list[dict] = []
    print(f"Chatting with {model} at {client.base_url}")
    print("Type 'quit' or Ctrl-C to exit.\n")

    first_turn = True
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        # Attach image only on the first turn
        img = image if first_turn else None
        content = build_content(user_input, img)
        history.append({"role": "user", "content": content})
        first_turn = False

        resp = client.chat.completions.create(
            model=model,
            messages=history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        print("\nAssistant: ", end="", flush=True)
        reply = ""
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print("\n")
        history.append({"role": "assistant", "content": reply})


def main():
    args = parse_args()
    if args.image is not None and not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    if args.prompt:
        one_shot(client, args.model, args.prompt, args.image,
                 args.max_tokens, args.temperature, args.top_p)
    else:
        interactive(client, args.model, args.image,
                    args.max_tokens, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
