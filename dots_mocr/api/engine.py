from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import tempfile
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from dots_mocr.api.datamodel.requests import (
    ConvertDocumentsOptions,
    ConvertDocumentsRequest,
    FileSourceRequest,
    HttpSourceRequest,
)
from dots_mocr.api.datamodel.responses import (
    ConvertDocumentResponse,
    ErrorItem,
    ExportDocumentResponse,
)

if TYPE_CHECKING:
    from dots_mocr.parser import DotsMOCRParser


def resolve_prompt_mode(options: ConvertDocumentsOptions) -> str:
    if options.prompt_mode:
        return options.prompt_mode
    return "prompt_layout_all_en" if options.do_ocr else "prompt_layout_only_en"


def _extract_plain_text(md: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", "", md)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"[*_]{1,3}(.*?)[*_]{1,3}", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$\n]+\$", "", text)
    text = re.sub(r"^\s*[-|]+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _assemble_outputs(
    results: list[dict], to_formats: list[str]
) -> tuple[Optional[str], Optional[list]]:
    md_parts: list[str] = []
    json_pages: list = []

    for r in results:
        if "md" in to_formats and r.get("md_content_path"):
            try:
                with open(r["md_content_path"], encoding="utf-8") as f:
                    md_parts.append(f.read())
            except OSError:
                pass

        if "json" in to_formats and r.get("layout_info_path"):
            try:
                with open(r["layout_info_path"], encoding="utf-8") as f:
                    json_pages.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                json_pages.append(None)

    md_content = "\n\n---\n\n".join(md_parts) if md_parts else None
    json_content = json_pages if json_pages else None
    return md_content, json_content


def _parse_pdf_with_page_range(
    parser: "DotsMOCRParser",
    file_path: str,
    stem: str,
    prompt_mode: str,
    save_dir: str,
    page_range: Optional[list[int]],
) -> list[dict]:
    from dots_mocr.utils.doc_utils import load_images_from_pdf

    if page_range is None:
        return parser.parse_pdf(file_path, stem, prompt_mode, save_dir)

    start_page, end_page = page_range[0], page_range[1]
    images = load_images_from_pdf(
        file_path, dpi=parser.dpi, start_page_id=start_page, end_page_id=end_page
    )
    if not images:
        return []

    tasks = [
        {
            "origin_image": img,
            "prompt_mode": prompt_mode,
            "save_dir": save_dir,
            "save_name": stem,
            "source": "pdf",
            "page_idx": start_page + i,
        }
        for i, img in enumerate(images)
    ]

    results = []
    with ThreadPool(min(len(tasks), parser.num_thread)) as pool:
        for result in pool.imap_unordered(
            lambda t: parser._parse_single_image(**t), tasks
        ):
            result["file_path"] = file_path
            results.append(result)
    results.sort(key=lambda x: x["page_no"])
    return results


def _convert_file_sync(
    parser: "DotsMOCRParser",
    file_path: str,
    filename: str,
    prompt_mode: str,
    page_range: Optional[list[int]],
    to_formats: list[str],
) -> ConvertDocumentResponse:
    start = time.monotonic()
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    tmp_out = tempfile.mkdtemp(prefix="mocr_out_")
    errors: list[ErrorItem] = []

    try:
        if suffix == ".pdf":
            results = _parse_pdf_with_page_range(
                parser, file_path, stem, prompt_mode, tmp_out, page_range
            )
        else:
            results = parser.parse_image(file_path, stem, prompt_mode, tmp_out)

        if not results:
            errors.append(ErrorItem(message="Parser returned no results"))
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename),
                status="failure",
                errors=errors,
                processing_time=time.monotonic() - start,
            )

        md_content, json_content = _assemble_outputs(results, to_formats)
        text_content = (
            _extract_plain_text(md_content)
            if md_content and "text" in to_formats
            else None
        )

        return ConvertDocumentResponse(
            document=ExportDocumentResponse(
                filename=filename,
                md_content=md_content if "md" in to_formats else None,
                json_content=json_content if "json" in to_formats else None,
                text_content=text_content,
            ),
            status="success",
            errors=[],
            processing_time=time.monotonic() - start,
        )
    except Exception as exc:
        errors.append(ErrorItem(message=str(exc)))
        return ConvertDocumentResponse(
            document=ExportDocumentResponse(filename=filename),
            status="failure",
            errors=errors,
            processing_time=time.monotonic() - start,
        )
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


async def _materialise_source(source: FileSourceRequest | HttpSourceRequest) -> tuple[str, str]:
    if isinstance(source, FileSourceRequest):
        suffix = Path(source.filename).suffix.lower()
        data = base64.b64decode(source.base64_string)
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path, source.filename

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(source.url, headers=source.headers)
        resp.raise_for_status()

    filename = Path(str(source.url)).name or "document.pdf"
    suffix = Path(filename).suffix.lower() or ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path, filename


async def convert_source(
    parser: "DotsMOCRParser",
    request: ConvertDocumentsRequest,
) -> list[ConvertDocumentResponse]:
    loop = asyncio.get_event_loop()
    prompt_mode = resolve_prompt_mode(request.options)
    results: list[ConvertDocumentResponse] = []

    for source in request.sources:
        file_path, filename = await _materialise_source(source)
        try:
            result = await loop.run_in_executor(
                None,
                _convert_file_sync,
                parser,
                file_path,
                filename,
                prompt_mode,
                request.options.page_range,
                request.options.to_formats,
            )
            results.append(result)
        finally:
            try:
                os.unlink(file_path)
            except OSError:
                pass

    return results
