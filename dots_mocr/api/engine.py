from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import tempfile
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

logger = logging.getLogger("uvicorn.error")

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


# GPU-bound work: cap how many documents are converted concurrently. Each
# document still fans out up to parser.num_thread page requests to vLLM, so
# effective GPU concurrency ≈ MOCR_MAX_CONCURRENT * num_thread.
_MAX_CONCURRENT = int(os.environ.get("MOCR_MAX_CONCURRENT", "2"))
_semaphore: Optional["asyncio.Semaphore"] = None


def _get_semaphore() -> "asyncio.Semaphore":
    # Created lazily so it binds to the running event loop.
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _semaphore


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
    image_mode: str = "base64",
    describe_script: Optional[str] = None,
) -> list[dict]:
    from dots_mocr.utils.doc_utils import load_images_from_pdf

    if page_range is None:
        return parser.parse_pdf(file_path, stem, prompt_mode, save_dir, image_mode=image_mode, describe_script=describe_script)

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
            "image_mode": image_mode,
            "describe_script": describe_script,
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
    image_mode: str = "base64",
    describe_script: Optional[str] = None,
) -> ConvertDocumentResponse:
    start = time.monotonic()
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    tmp_out = tempfile.mkdtemp(prefix="mocr_out_")
    errors: list[ErrorItem] = []
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else -1
    logger.info(
        "convert start: filename=%s suffix=%s size=%dB prompt_mode=%s "
        "page_range=%s to_formats=%s image_mode=%s tmp_out=%s",
        filename, suffix, file_size, prompt_mode, page_range, to_formats,
        image_mode, tmp_out,
    )

    try:
        if suffix == ".pdf":
            results = _parse_pdf_with_page_range(
                parser, file_path, stem, prompt_mode, tmp_out, page_range,
                image_mode=image_mode, describe_script=describe_script,
            )
        else:
            results = parser.parse_image(file_path, stem, prompt_mode, tmp_out, image_mode=image_mode, describe_script=describe_script)

        if not results:
            elapsed = time.monotonic() - start
            logger.warning(
                "convert failure: filename=%s no results (0 renderable pages) "
                "in %.2fs",
                filename, elapsed,
            )
            errors.append(ErrorItem(message="Parser returned no results (0 renderable pages)"))
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename),
                status="failure",
                errors=errors,
                processing_time=elapsed,
            )

        md_content, json_content = _assemble_outputs(results, to_formats)
        text_content = (
            _extract_plain_text(md_content)
            if md_content and "text" in to_formats
            else None
        )

        elapsed = time.monotonic() - start
        logger.info(
            "convert success: filename=%s pages=%d md_chars=%s json_pages=%s "
            "in %.2fs",
            filename, len(results),
            len(md_content) if md_content else 0,
            len(json_content) if json_content else 0,
            elapsed,
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
            processing_time=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.exception(
            "convert error: filename=%s failed after %.2fs: %s",
            filename, elapsed, exc,
        )
        errors.append(ErrorItem(message=str(exc)))
        return ConvertDocumentResponse(
            document=ExportDocumentResponse(filename=filename),
            status="failure",
            errors=errors,
            processing_time=elapsed,
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
        logger.debug(
            "materialised upload: filename=%s size=%dB -> %s",
            source.filename, len(data), path,
        )
        return path, source.filename

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(source.url, headers=source.headers)
        resp.raise_for_status()

    filename = Path(str(source.url)).name or "document.pdf"
    suffix = Path(filename).suffix.lower() or ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    logger.debug(
        "materialised url: %s -> filename=%s size=%dB -> %s",
        source.url, filename, len(resp.content), path,
    )
    return path, filename


async def convert_source(
    parser: "DotsMOCRParser",
    request: ConvertDocumentsRequest,
) -> list[ConvertDocumentResponse]:
    loop = asyncio.get_event_loop()
    prompt_mode = resolve_prompt_mode(request.options)
    semaphore = _get_semaphore()
    results: list[ConvertDocumentResponse] = []

    for source in request.sources:
        file_path, filename = await _materialise_source(source)
        try:
            if semaphore.locked():
                logger.info(
                    "convert queued: filename=%s waiting for a free slot "
                    "(max_concurrent=%d)",
                    filename, _MAX_CONCURRENT,
                )
            async with semaphore:
                result = await loop.run_in_executor(
                    None,
                    _convert_file_sync,
                    parser,
                    file_path,
                    filename,
                    prompt_mode,
                    request.options.page_range,
                    request.options.to_formats,
                    request.options.image_mode,
                    request.options.describe_script,
                )
            results.append(result)
        finally:
            try:
                os.unlink(file_path)
            except OSError:
                pass

    return results
