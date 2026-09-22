import json
import time

import pytest
from PIL import Image

from dots_mocr import parser as parser_mod
from dots_mocr.parser import DotsMOCRParser
from dots_mocr.utils.prompts import dict_promptmode_to_fallback_prompt, dict_promptmode_to_prompt
from tests.conftest import api_status_error, connection_error


class _Backend:
    """Scripted inference_with_vllm: per host, a result or an exception."""

    def __init__(self):
        self.behaviour = {"main": "main-answer", "fb": "fb-answer"}
        self.calls = []
        self.prompts = []

    def __call__(self, image, prompt, *, ip, **kwargs):
        self.calls.append((ip, kwargs.get("strip_reasoning", False)))
        self.prompts.append((ip, prompt))
        outcome = self.behaviour[ip]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def backend(monkeypatch):
    b = _Backend()
    monkeypatch.setattr(parser_mod, "inference_with_vllm", b)
    return b


def _parser(cooldown=60.0, **fallback):
    return DotsMOCRParser(
        ip="main",
        fallback={"ip": "fb", "model_name": "fb-model", **fallback},
        fallback_cooldown=cooldown,
    )


def _call(p):
    return p._inference_with_vllm(None, "prompt", "prompt_layout_all_en")


def test_main_ok(backend):
    p = _parser()
    assert _call(p) == ("main-answer", False)
    assert backend.calls == [("main", False)]


def test_no_fallback_configured_propagates(backend):
    backend.behaviour["main"] = connection_error()
    p = DotsMOCRParser(ip="main")
    with pytest.raises(type(connection_error())):
        _call(p)


def test_connection_error_falls_back_and_arms_breaker(backend):
    backend.behaviour["main"] = connection_error()
    p = _parser()
    assert _call(p) == ("fb-answer", True)
    assert p._main_down_until > time.monotonic()

    # While armed, main is skipped entirely.
    backend.calls.clear()
    assert _call(p) == ("fb-answer", True)
    assert backend.calls == [("fb", True)]


def test_client_error_neither_falls_back_nor_arms(backend):
    backend.behaviour["main"] = api_status_error(400)
    p = _parser()
    with pytest.raises(type(api_status_error(400))):
        _call(p)
    assert backend.calls == [("main", False)]
    assert p._main_down_until == 0.0


def test_server_error_falls_back_without_arming(backend):
    backend.behaviour["main"] = api_status_error(500)
    p = _parser()
    assert _call(p) == ("fb-answer", True)
    assert p._main_down_until == 0.0


def test_cooldown_zero_never_arms(backend):
    backend.behaviour["main"] = connection_error()
    p = _parser(cooldown=0)
    assert _call(p) == ("fb-answer", True)
    assert p._main_down_until == 0.0


def test_both_down_raises_fallback_error_chained(backend):
    main_exc = connection_error()
    backend.behaviour["main"] = main_exc
    backend.behaviour["fb"] = api_status_error(503)
    p = _parser()
    with pytest.raises(type(api_status_error(503))) as info:
        _call(p)
    assert info.value.__cause__ is main_exc


def test_breaker_probes_main_when_fallback_fails(backend):
    p = _parser()
    p._main_down_until = time.monotonic() + 60
    backend.behaviour["fb"] = connection_error()
    assert _call(p) == ("main-answer", False)
    assert p._main_down_until == 0.0
    assert [ip for ip, _ in backend.calls] == ["fb", "main"]


def test_breaker_probe_fails_too(backend):
    p = _parser()
    p._main_down_until = time.monotonic() + 60
    fb_exc = connection_error()
    backend.behaviour["fb"] = fb_exc
    backend.behaviour["main"] = connection_error()
    with pytest.raises(type(fb_exc)) as info:
        _call(p)
    assert info.value.__cause__ is fb_exc
    assert p._main_down_until > time.monotonic()


def test_strip_thinking_can_be_disabled(backend):
    backend.behaviour["main"] = connection_error()
    p = _parser(strip_thinking=False)
    _call(p)
    assert backend.calls[-1] == ("fb", False)


def _parse_png(p, tmp_path):
    src = tmp_path / "page.png"
    Image.new("RGB", (644, 924), "white").save(src)
    out = tmp_path / "out"
    out.mkdir()
    return p.parse_image(str(src), "page", "prompt_layout_all_en", str(out))[0]


def test_unrecoverable_answer_fails_page(backend, tmp_path):
    backend.behaviour["main"] = connection_error()
    backend.behaviour["fb"] = "I cannot see any document in this image."
    result = _parse_png(_parser(), tmp_path)
    assert result["error"]["code"] == "page_failed"
    assert "fb-model" in result["error"]["message"]


def test_qwen_style_fallback_answer_parses_and_scales(backend, tmp_path):
    backend.behaviour["main"] = connection_error()
    backend.behaviour["fb"] = (
        '```json\n[{"bbox_2d": [0, 0, 500, 500], "text_content": "Appendix D"}]\n```'
    )
    result = _parse_png(_parser(bbox_scale=1000), tmp_path)
    assert "error" not in result and not result.get("filtered")
    assert result["fallback_model"] == "fb-model"
    with open(result["md_content_path"], encoding="utf-8") as f:
        assert "Appendix D" in f.read()
    with open(result["layout_info_path"], encoding="utf-8") as f:
        assert json.load(f)[0]["bbox"] == [0, 0, 322, 462]


LAYOUT = "prompt_layout_all_en"


def test_fallback_gets_dots_prompt_by_default(backend):
    backend.behaviour["main"] = connection_error()
    _parser()._inference_with_vllm(None, dict_promptmode_to_prompt[LAYOUT], LAYOUT)
    assert backend.prompts[-1] == ("fb", dict_promptmode_to_prompt[LAYOUT])


def test_generic_prompts_only_for_fallback(backend):
    backend.behaviour["main"] = connection_error()
    _parser(prompts="generic")._inference_with_vllm(None, dict_promptmode_to_prompt[LAYOUT], LAYOUT)
    assert backend.prompts == [
        ("main", dict_promptmode_to_prompt[LAYOUT]),
        ("fb", dict_promptmode_to_fallback_prompt[LAYOUT]),
    ]


def test_generic_prompts_keep_non_stock_prompt(backend):
    backend.behaviour["main"] = connection_error()
    p = _parser(prompts="generic")
    grounding = dict_promptmode_to_prompt["prompt_grounding_ocr"] + "[1, 2, 3, 4]"
    p._inference_with_vllm(None, grounding, "prompt_grounding_ocr")
    p._inference_with_vllm(None, "custom", LAYOUT)
    assert [pr for ip, pr in backend.prompts if ip == "fb"] == [grounding, "custom"]
