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


class ErrorItem(BaseModel):
    message: str


class ConvertDocumentResponse(BaseModel):
    document: ExportDocumentResponse
    status: str
    errors: list[ErrorItem] = []
    processing_time: float


class TaskStatusResponse(BaseModel):
    task_id: str
    task_status: str
    task_position: Optional[int] = None
    error_message: Optional[str] = None


class VersionResponse(BaseModel):
    name: str = "dots.mocr-serve"
    version: str = "1.0.0"
    docling_serve_compat: str = "0.1"
