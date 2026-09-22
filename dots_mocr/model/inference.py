import os
import re
import threading
import time

import openai
from openai import OpenAI

from dots_mocr.log import logger
from dots_mocr.utils.image_utils import PILimage_to_base64

# Per-request timeout for vLLM inference calls. Without one, the OpenAI SDK
# default (600s) lets a wedged vLLM pin ThreadPool threads — and the API's
# concurrency slots — for 10 minutes per page.
_VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "300"))

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def strip_thinking(text):
    """Remove reasoning a thinking model (e.g. the fallback) put into its answer.

    Handles full ``<think>…</think>`` blocks, an orphan ``</think>`` (the chat
    template injected the opening tag) and an unclosed ``<think>`` (generation
    cut off mid-thought, leaving no answer). Text without tags is returned as is.
    """
    if not text:
        return text
    if not (_THINK_OPEN.search(text) or _THINK_CLOSE.search(text)):
        return text
    text = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opened = _THINK_OPEN.search(text)
    if opened:
        text = text[:opened.start()]
    return text.strip()


class ModelOutputError(RuntimeError):
    """The model answered, but with nothing usable (e.g. reasoning only).

    Not an ``openai.APIError``: the backend is fine, so there is no fallback
    retry and the page is reported as ``page_failed``.
    """


_clients: dict[tuple[str, str], OpenAI] = {}
_clients_lock = threading.Lock()


def _get_client(addr: str, api_key: str) -> OpenAI:
    with _clients_lock:
        client = _clients.get((addr, api_key))
        if client is None:
            client = OpenAI(
                api_key=api_key,
                base_url=addr,
                timeout=_VLLM_TIMEOUT,
            )
            _clients[(addr, api_key)] = client
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
        api_key=None,
        strip_reasoning=False,
        ):

    addr = f"{protocol}://{ip}:{port}/v1"
    client = _get_client(addr, api_key or os.environ.get("API_KEY", "0"))
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
    choice = response.choices[0]
    content = choice.message.content
    elapsed = time.monotonic() - start
    if choice.finish_reason == "length":
        logger.warning(
            "vllm response truncated at max_tokens={} (model={}, len={})",
            max_completion_tokens, model_name, len(content or ""),
        )
    # Only the answer is used; reasoning_content (vLLM --reasoning-parser) is
    # deliberately ignored. Inline thoughts are stripped only when asked: a
    # document may legitimately contain the literal text "<think>".
    if strip_reasoning and content:
        raw = content
        content = strip_thinking(raw)
        if len(content) != len(raw):
            logger.debug("stripped thinking from response: {} chars removed", len(raw) - len(content))
        if not content:
            raise ModelOutputError(
                f"model {model_name} returned reasoning but no answer "
                f"(finish_reason={choice.finish_reason})"
            )
    if not content:
        logger.warning("vllm response empty (model={}) in {:.2f}s", model_name, elapsed)
    else:
        logger.debug("vllm response: len={} in {:.2f}s", len(content), elapsed)
    return content


def is_upstream_error(exc: BaseException) -> bool:
    """True when the failure is the vLLM backend's, not the request's.

    Connection refused, read timeouts, 5xx and 429. These are retried on the
    fallback model and mapped to 502, so a client can tell "the model is down"
    from "we broke". 4xx (prompt too long, bad image, unknown model) is not: the
    same request would fail anywhere, and it must not trip the breaker.
    """
    return isinstance(
        exc,
        (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError),
    )


def is_backend_down(exc: BaseException) -> bool:
    """True when the backend is unreachable (includes timeouts).

    Only these trip the fallback breaker; a single 5xx may be specific to one
    request and should not reroute every page.
    """
    return isinstance(exc, openai.APIConnectionError)
