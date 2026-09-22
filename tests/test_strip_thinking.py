import pytest

from dots_mocr.model.inference import strip_thinking


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("plain answer", "plain answer"),
        ("", ""),
        (None, None),
        ("<think>reasoning</think>answer", "answer"),
        ("<think>\nmulti\nline\n</think>\n\n  answer  ", "answer"),
        ("<THINK>shouting</Think>answer", "answer"),
        ("<think>a</think>x<think>b</think>y", "xy"),
        # Chat template injected the opening tag: only the close is visible.
        ("reasoning without open tag</think>answer", "answer"),
        # Cut off mid-thought: nothing usable.
        ("<think>still thinking when max_tokens hit", ""),
        ("answer<think>trailing unclosed", "answer"),
    ],
)
def test_strip_thinking(raw, expected):
    assert strip_thinking(raw) == expected


def test_text_without_tags_is_not_stripped_of_whitespace():
    assert strip_thinking("  keep  \n") == "  keep  \n"
