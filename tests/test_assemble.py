import json

import pytest

from dots_mocr.api.engine import _assemble_outputs


@pytest.fixture
def page(tmp_path):
    def make(page_no, md=None, layout=None, **extra):
        r = {"page_no": page_no, **extra}
        if md is not None:
            path = tmp_path / f"p{page_no}.md"
            path.write_text(md, encoding="utf-8")
            r["md_content_path"] = str(path)
        if layout is not None:
            path = tmp_path / f"p{page_no}.json"
            path.write_text(json.dumps(layout), encoding="utf-8")
            r["layout_info_path"] = str(path)
        return r
    return make


def _failed(page_no, code="page_failed"):
    return {"page_no": page_no, "error": {"code": code, "message": "boom"}}


def test_failed_page_keeps_its_slot(page):
    results = [page(0, "zero", [{"c": 0}]), _failed(1), page(2, "two", [{"c": 2}])]
    md, js, errors, ok = _assemble_outputs(results, ["md", "json"])

    assert ok == 2
    assert js == [[{"c": 0}], None, [{"c": 2}]]
    assert md.split("\n\n---\n\n") == [
        "zero", "<!-- dots.mocr: page 1 failed (page_failed) -->", "two",
    ]
    assert [(e.code, e.page_no) for e in errors] == [("page_failed", 1)]


def test_all_pages_failed_has_no_content(page):
    md, js, errors, ok = _assemble_outputs(
        [_failed(0), _failed(1, "page_model_error")], ["md", "json"]
    )
    assert (md, js, ok) == (None, [None, None], 0)
    assert [e.code for e in errors] == ["page_failed", "page_model_error"]


def test_pages_without_json_do_not_produce_a_json_list(page):
    md, js, _, ok = _assemble_outputs([page(0, "svg page")], ["md", "json"])
    assert md == "svg page"
    assert js is None
    assert ok == 1


def test_diagnostics_are_reported_but_page_counts_as_ok(page):
    results = [
        page(0, "a", filtered=True),
        page(1, "", empty_response=True),
        page(2, "c", fallback_model="fb-model"),
    ]
    _, _, errors, ok = _assemble_outputs(results, ["md"])
    assert ok == 3
    assert [(e.code, e.page_no) for e in errors] == [
        ("page_degraded", 0), ("page_empty_response", 1), ("page_fallback_model", 2),
    ]
    assert "fb-model" in errors[2].message


def test_text_only_still_reads_markdown(page):
    md, js, _, _ = _assemble_outputs([page(0, "# Title", [{"c": 0}])], ["text"])
    assert md == "# Title"
    assert js is None


def test_json_only_skips_markdown(page):
    md, js, _, _ = _assemble_outputs([page(0, "# Title", [{"c": 0}])], ["json"])
    assert md is None
    assert js == [[{"c": 0}]]
