from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

VALID_PROMPT_MODES = {
    "prompt_layout_all_en",
    "prompt_layout_only_en",
    "prompt_ocr",
    "prompt_web_parsing",
    "prompt_scene_spotting",
    "prompt_image_to_svg",
    "prompt_general",
    "prompt_grounding_ocr",
}

VALID_TO_FORMATS = {"md", "json", "text"}


class FileSourceRequest(BaseModel):
    kind: Literal["file"] = "file"
    base64_string: str
    filename: str

    @field_validator("filename")
    @classmethod
    def check_extension(cls, v: str) -> str:
        from pathlib import Path
        ext = Path(v).suffix.lower()
        if ext not in VALID_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported: {VALID_EXTENSIONS}")
        return v


class HttpSourceRequest(BaseModel):
    kind: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = {}


SourceRequestItem = Annotated[
    Union[FileSourceRequest, HttpSourceRequest],
    Field(discriminator="kind"),
]


class ConvertDocumentsOptions(BaseModel):
    do_ocr: bool = True
    to_formats: list[str] = ["md", "json"]
    page_range: Optional[list[int]] = None
    prompt_mode: Optional[str] = None

    @field_validator("prompt_mode")
    @classmethod
    def check_prompt_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_PROMPT_MODES:
            raise ValueError(f"Unknown prompt_mode '{v}'. Valid: {sorted(VALID_PROMPT_MODES)}")
        return v

    @field_validator("page_range")
    @classmethod
    def check_page_range(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None:
            if len(v) != 2:
                raise ValueError("page_range must be [start, end] (0-indexed, inclusive)")
            if v[0] < 0 or v[1] < v[0]:
                raise ValueError("page_range must satisfy 0 <= start <= end")
        return v

    @field_validator("to_formats")
    @classmethod
    def check_to_formats(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_TO_FORMATS
        if invalid:
            raise ValueError(f"Unknown to_formats values: {invalid}. Valid: {VALID_TO_FORMATS}")
        return v


class ConvertDocumentsRequest(BaseModel):
    sources: list[SourceRequestItem]
    options: ConvertDocumentsOptions = Field(default_factory=ConvertDocumentsOptions)
