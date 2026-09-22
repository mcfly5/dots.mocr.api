import pytest
from fastapi import HTTPException

from dots_mocr.api.datamodel.responses import (
    ConvertDocumentResponse,
    ErrorItem,
    ExportDocumentResponse,
)
from dots_mocr.api.engine import enforce_failure_policy


def _doc(name, status, *codes):
    return ConvertDocumentResponse(
        document=ExportDocumentResponse(filename=name),
        status=status,
        errors=[ErrorItem(message=f"{c} happened", code=c, page_no=i) for i, c in enumerate(codes)],
        processing_time=1.0,
    )


OK = _doc("ok.pdf", "success")
OK_WITH_DIAGNOSTIC = _doc("diag.pdf", "success", "page_fallback_model")
PARTIAL = _doc("partial.pdf", "partial_success", "page_failed")
MODEL_DOWN = _doc("down.pdf", "failure", "page_model_error", "page_model_error")
INVALID = _doc("bad.png", "failure", "document_invalid")
BROKEN = _doc("broken.pdf", "failure", "page_failed")


def _status(results, allow_partial):
    try:
        enforce_failure_policy(results, allow_partial)
    except HTTPException as exc:
        return exc.status_code
    return 200


@pytest.mark.parametrize(
    "results, allow_partial, expected",
    [
        ([], False, 200),
        ([OK, OK_WITH_DIAGNOSTIC], False, 200),
        ([PARTIAL], False, 500),
        ([PARTIAL], True, 200),
        ([OK, BROKEN], False, 500),
        ([OK, BROKEN], True, 200),
        ([MODEL_DOWN], False, 502),
        ([MODEL_DOWN], True, 502),
        ([MODEL_DOWN, BROKEN], True, 500),
        ([INVALID], False, 422),
        ([INVALID], True, 422),
        ([INVALID, MODEL_DOWN], True, 500),
    ],
)
def test_status_codes(results, allow_partial, expected):
    assert _status(results, allow_partial) == expected


def test_error_body_lists_every_document_with_page_errors():
    with pytest.raises(HTTPException) as info:
        enforce_failure_policy([OK, PARTIAL], allow_partial=False)
    detail = info.value.detail
    assert detail["message"].startswith("1 of 2 document(s) failed")
    assert "partial.pdf" in detail["message"]
    assert [d["filename"] for d in detail["documents"]] == ["ok.pdf", "partial.pdf"]
    assert detail["documents"][1]["errors"] == [
        {"message": "page_failed happened", "code": "page_failed", "page_no": 0}
    ]
