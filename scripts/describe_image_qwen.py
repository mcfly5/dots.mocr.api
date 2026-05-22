#!/usr/bin/env python3
"""
Describe a cropped image using a locally deployed Qwen model via vLLM.

Usage:
    python scripts/describe_image_qwen.py <image_path>

All config via LLM_DESCRIPTION_* environment variables:
    LLM_DESCRIPTION_HOST       (default: localhost)
    LLM_DESCRIPTION_PORT       (default: 8000)
    LLM_DESCRIPTION_PROTOCOL   (default: http)
    LLM_DESCRIPTION_MODEL      (required)
    LLM_DESCRIPTION_API_KEY    (default: EMPTY)
    LLM_DESCRIPTION_PROMPT     (optional override)
    LLM_DESCRIPTION_MAX_TOKENS (default: 256)
"""
import sys
import os
import base64
from io import BytesIO

try:
    from openai import OpenAI
    from PIL import Image
except ImportError as e:
    print(f"[missing dependency: {e}]", file=sys.stderr)
    print("[Image]")
    sys.exit(1)


_env = os.environ.get
DEFAULT_PROMPT = "Briefly describe what is shown in this image in one concise sentence."
MAX_IMAGE_SIDE = 512  # resize longest side to this before encoding


def image_to_base64_uri(path: str) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def describe(image_path: str) -> str:
    protocol = _env("LLM_DESCRIPTION_PROTOCOL", "http")
    host     = _env("LLM_DESCRIPTION_HOST", "localhost")
    port     = _env("LLM_DESCRIPTION_PORT", "8000")
    model    = _env("LLM_DESCRIPTION_MODEL")
    api_key  = _env("LLM_DESCRIPTION_API_KEY", "EMPTY")
    prompt   = _env("LLM_DESCRIPTION_PROMPT", DEFAULT_PROMPT)
    max_tok  = int(_env("LLM_DESCRIPTION_MAX_TOKENS", "2000"))

    if not model:
        raise ValueError("LLM_DESCRIPTION_MODEL env var is required")

    client = OpenAI(
        api_key=api_key,
        base_url=f"{protocol}://{host}:{port}/v1",
    )
    data_uri = image_to_base64_uri(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_completion_tokens=max_tok,
        temperature=0.1,
        top_p=0.9,
    )
    msg = response.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", "") or "[Image]"
    # strip <think>...</think> blocks
    import re as _re
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    # take only the first sentence as a safety net against verbose reasoning output
    first = _re.split(r"(?<=[.!?])\s", text)[0]
    return (first or text).replace("\n", " ") or "[Image]"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Image]")
        sys.exit(0)

    try:
        print(describe(sys.argv[1]))
    except Exception as e:
        print(f"[vllm error: {e}]", file=sys.stderr)
        print("[Image]")
