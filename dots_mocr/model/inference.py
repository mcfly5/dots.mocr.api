import os
import threading
import time

from openai import OpenAI

from dots_mocr.log import logger
from dots_mocr.utils.image_utils import PILimage_to_base64

# Per-request timeout for vLLM inference calls. Without one, the OpenAI SDK
# default (600s) lets a wedged vLLM pin ThreadPool threads — and the API's
# concurrency slots — for 10 minutes per page.
_VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "300"))

_clients: dict[str, OpenAI] = {}
_clients_lock = threading.Lock()


def _get_client(addr: str) -> OpenAI:
    with _clients_lock:
        client = _clients.get(addr)
        if client is None:
            client = OpenAI(
                api_key=os.environ.get("API_KEY", "0"),
                base_url=addr,
                timeout=_VLLM_TIMEOUT,
            )
            _clients[addr] = client
        return client


def inference_with_vllm(
        image,
        prompt,
        protocol="http",
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='rednote-hilab/dots.mocr',
        system_prompt=None,
        ):

    addr = f"{protocol}://{ip}:{port}/v1"
    client = _get_client(addr)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    logger.debug(
        "vllm request: addr={} model={} temperature={} top_p={} max_tokens={} "
        "image={}x{} prompt_len={}",
        addr, model_name, temperature, top_p, max_completion_tokens,
        getattr(image, "width", "?"), getattr(image, "height", "?"), len(prompt),
    )
    start = time.monotonic()
    # Deliberately unguarded: failures propagate to the caller so the API layer
    # (dots_mocr/api/engine.py) can log and surface them, rather than turning a
    # wedged vLLM into a silently empty page.
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p)
    content = response.choices[0].message.content
    elapsed = time.monotonic() - start
    if not content:
        logger.warning("vllm response empty (model={}) in {:.2f}s", model_name, elapsed)
    else:
        logger.debug("vllm response: len={} in {:.2f}s", len(content), elapsed)
    return content
