import importlib
import sys
import types

import httpx
import openai
import pytest

# cairosvg needs the system libcairo, which is only present in the server image.
# Nothing under test renders SVG, so a stub keeps the package importable anywhere.
try:
    importlib.import_module("cairosvg")
except OSError:
    sys.modules["cairosvg"] = types.ModuleType("cairosvg")


_REQUEST = httpx.Request("POST", "http://vllm.test/v1/chat/completions")


def api_status_error(status_code: int) -> openai.APIStatusError:
    cls = {
        400: openai.BadRequestError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
    }.get(status_code, openai.InternalServerError)
    return cls(
        f"status {status_code}",
        response=httpx.Response(status_code, request=_REQUEST),
        body=None,
    )


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_REQUEST)


@pytest.fixture(autouse=True)
def _reset_engine_semaphore():
    # The engine's semaphore binds to the first event loop that uses it; each
    # TestClient runs its own loop.
    from dots_mocr.api import engine
    engine._semaphore = None
    yield
    engine._semaphore = None
