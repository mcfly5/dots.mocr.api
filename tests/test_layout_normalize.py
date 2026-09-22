import json

import pytest
from PIL import Image

from dots_mocr.utils.layout_utils import markdown_table_to_html, normalize_layout_response, post_process_output

# Shape of a Qwen3.5 fallback answer to prompt_layout_all_en.
QWEN_ANSWER = (
    '```json\n[\n'
    '\t{"bbox_2d": [439, 66, 558, 101], "text_content": "Appendix D"},\n'
    '\t{"bbox_2d": [439, 106, 558, 130], "text_content": "Milk"}\n'
    ']\n```'
)
DOTS_ANSWER = '[{"bbox": [10, 20, 100, 40], "category": "Title", "text": "# Appendix D"}]'


@pytest.fixture
def images():
    # 644x924 is already a smart_resize fixed point, so pixel boxes map 1:1.
    img = Image.new("RGB", (644, 924), "white")
    return img, img


def test_dots_answer_unchanged():
    assert normalize_layout_response(DOTS_ANSWER) == DOTS_ANSWER


def test_strips_fence_and_renames_keys():
    out = normalize_layout_response(QWEN_ANSWER)
    assert out.startswith("[") and out.endswith("]")
    assert '"bbox": [439' in out and '"text": "Appendix D"' in out
    assert "bbox_2d" not in out and "text_content" not in out


def test_unterminated_fence_is_stripped():
    assert normalize_layout_response("```json\n[]").strip() == "[]"


def test_qwen_answer_parses_with_default_category(images):
    cells, filtered = post_process_output(QWEN_ANSWER, "prompt_layout_all_en", *images)
    assert not filtered
    assert [c["text"] for c in cells] == ["Appendix D", "Milk"]
    assert all(c["category"] == "Text" for c in cells)
    assert cells[0]["bbox"] == [439, 66, 558, 101]


def test_relative_bbox_scaled_to_pixels(images):
    cells, filtered = post_process_output(
        QWEN_ANSWER, "prompt_layout_all_en", *images, bbox_scale=1000
    )
    assert not filtered
    assert cells[0]["bbox"] == [int(439 * 0.644), int(66 * 0.924), int(558 * 0.644), int(101 * 0.924)]


def test_truncated_qwen_answer_salvages_text(images):
    truncated = QWEN_ANSWER[: QWEN_ANSWER.index('\t{"bbox_2d": [439, 106')] + '\t{"bbox_2d": [439, 1'
    text, filtered = post_process_output(truncated, "prompt_layout_all_en", *images)
    assert filtered
    assert "Appendix D" in text


def test_markdown_table_converted_to_html(images):
    answer = json.dumps([{
        "bbox_2d": [0, 0, 100, 100], "category": "Table",
        "text": "| Sample | Fat, % |\n|---|:--:|\n| A | 3.2 |\n| B<1 | 2.5 |",
    }])
    cells, _ = post_process_output(answer, "prompt_layout_all_en", *images)
    assert cells[0]["text"] == (
        "<table><tr><td>Sample</td><td>Fat, %</td></tr>"
        "<tr><td>A</td><td>3.2</td></tr><tr><td>B&lt;1</td><td>2.5</td></tr></table>"
    )


@pytest.mark.parametrize("text", [
    "<table><tr><td>A</td></tr></table>",  # already HTML
    "Sample  Fat\nA  3.2",                   # plain text: no reliable columns
    "a | b",                                 # a pipe, but no separator row
])
def test_non_markdown_table_left_alone(text):
    assert markdown_table_to_html(text) is None
