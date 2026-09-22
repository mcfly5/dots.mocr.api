from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ExportDocumentResponse(BaseModel):
    filename: str
    md_content: Optional[str] = None
    json_content: Optional[Any] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    doctags_content: Optional[str] = None


# Fatal codes make a document fail (or, with allow_partial_results, degrade to
# "partial_success"); diagnostic ones are reported but never change the status.
FATAL_ERROR_CODES = {
    "page_failed", "page_model_error", "page_skipped", "document_failed",
    "document_invalid",
}


class ErrorItem(BaseModel):
    message: str
    # page_failed | page_model_error | page_skipped | page_empty_response |
    # page_degraded | page_fallback_model | document_failed | document_invalid
    code: Optional[str] = None
    # 0-indexed PDF page number; None for document-level errors.
    page_no: Optional[int] = None


class ConvertDocumentResponse(BaseModel):
    document: ExportDocumentResponse
    # "success" | "partial_success" | "failure"
    status: str
    errors: list[ErrorItem] = []
    processing_time: float


class DocumentErrorSummary(BaseModel):
    """One document inside an HTTP error body: its errors, none of its content."""

    filename: str
    status: str
    errors: list[ErrorItem] = []
    processing_time: float


class ConvertFailureDetail(BaseModel):
    """Body of the 500/502 returned when a conversion failed.

    Sent as FastAPI's ``detail``, so the per-page error structure survives even
    though the response is not a ConvertDocumentResponse list.
    """

    message: str
    documents: list[DocumentErrorSummary] = []


class TaskStatusResponse(BaseModel):
    task_id: str
    task_status: str
    task_position: Optional[int] = None
    error_message: Optional[str] = None


class VersionResponse(BaseModel):
    name: str = "dots.mocr-serve"
    version: str = "1.0.0"
    docling_serve_compat: str = "0.1"
