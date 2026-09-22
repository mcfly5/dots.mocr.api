from types import SimpleNamespace

import pytest
from PIL import Image

from dots_mocr.model import inference
from dots_mocr.model.inference import (
    ModelOutputError,
    inference_with_vllm,
    is_backend_down,
    is_upstream_error,
)
from tests.conftest import api_status_error, connection_error, timeout_error


@pytest.mark.parametrize(
    "exc, upstream, down",
    [
        (connection_error(), True, True),
        (timeout_error(), True, True),
        (api_status_error(500), True, False),
        (api_status_error(503), True, False),
        (api_status_error(429), True, False),
        (api_status_error(400), False, False),
        (api_status_error(404), False, False),
        (ValueError("ours"), False, False),
    ],
)
def test_error_classification(exc, upstream, down):
    assert is_upstream_error(exc) is upstream
    assert is_backend_down(exc) is down


class _FakeClient:
    def __init__(self, content, finish_reason="stop"):
        self.calls = 0
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )

        def create(**kwargs):
            self.calls += 1
            return reply

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.fixture
def fake_client(monkeypatch):
    def install(content, finish_reason="stop"):
        client = _FakeClient(content, finish_reason)
        monkeypatch.setattr(inference, "_get_client", lambda addr, key: client)
        return client
    return install


_IMG = Image.new("RGB", (8, 8))


def test_main_model_output_is_not_stripped(fake_client):
    fake_client("a paper about <think> tokens")
    assert inference_with_vllm(_IMG, "p") == "a paper about <think> tokens"


def test_strip_reasoning(fake_client):
    fake_client("<think>hmm</think>answer")
    assert inference_with_vllm(_IMG, "p", strip_reasoning=True) == "answer"


def test_reasoning_only_answer_raises(fake_client):
    fake_client("<think>ran out of tokens", finish_reason="length")
    with pytest.raises(ModelOutputError, match="finish_reason=length"):
        inference_with_vllm(_IMG, "p", strip_reasoning=True)


def test_empty_answer_is_returned_not_raised(fake_client):
    fake_client("")
    assert inference_with_vllm(_IMG, "p", strip_reasoning=True) == ""


def test_truncated_answer_is_still_returned(fake_client):
    fake_client('[{"bbox": [1, 2', finish_reason="length")
    assert inference_with_vllm(_IMG, "p") == '[{"bbox": [1, 2'
