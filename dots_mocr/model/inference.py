import os
import threading

from openai import OpenAI

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
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p)
    return response.choices[0].message.content
