from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

logger = logging.getLogger("uvicorn.error")

from dots_mocr.api.datamodel.requests import (
    VALID_EXTENSIONS,
    ConvertDocumentsOptions,
    ConvertDocumentsRequest,
    FileSourceRequest,
    HttpSourceRequest,
)
from dots_mocr.api.datamodel.responses import (
    FATAL_ERROR_CODES,
    ConvertDocumentResponse,
    ConvertFailureDetail,
    DocumentErrorSummary,
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


_FAILED_PAGE_MD = "<!-- dots.mocr: page {page_no} failed ({code}) -->"

# Distinguishes "this page has no json at all" (e.g. an SVG prompt mode) from
# "this page's json is missing because it failed", which is serialised as null.
_NO_JSON = object()


def _assemble_outputs(
    results: list[dict], to_formats: list[str]
) -> tuple[Optional[str], Optional[list], list[ErrorItem], int]:
    """Join the per-page artifacts and report what went wrong on the way.

    Failed pages keep their slot — a marker in the markdown, ``null`` in the json
    list — so ``json_content[i]`` still lines up with the page carrying
    ``errors[].page_no``.
    """
    md_slots: list[Optional[str]] = []
    json_slots: list = []
    errors: list[ErrorItem] = []
    pages_ok = 0
    md_from_page = False

    for r in results:
        page_no = r.get("page_no")
        error = r.get("error")

        if error:
            code = error.get("code") or "page_failed"
            errors.append(
                ErrorItem(
                    message=error.get("message") or "page processing failed",
                    code=code,
                    page_no=page_no,
                )
            )
            md_slots.append(_FAILED_PAGE_MD.format(page_no=page_no, code=code))
            json_slots.append(None)
            continue

        pages_ok += 1
        if r.get("filtered"):
            errors.append(
                ErrorItem(
                    message=(
                        "layout json could not be parsed; text recovered by the "
                        "fallback cleaner, no layout for this page"
                    ),
                    code="page_degraded",
                    page_no=page_no,
                )
            )
        if r.get("empty_response"):
            errors.append(
                ErrorItem(
                    message="the model returned an empty response for this page",
                    code="page_empty_response",
                    page_no=page_no,
                )
            )
        if r.get("fallback_model"):
            errors.append(
                ErrorItem(
                    message=(
                        "main model unavailable; page processed by fallback "
                        f"model {r['fallback_model']}"
                    ),
                    code="page_fallback_model",
                    page_no=page_no,
                )
            )

        md_path = r.get("md_content_path") if "md" in to_formats else None
        if md_path:
            try:
                with open(md_path, encoding="utf-8") as f:
                    md_slots.append(f.read())
                md_from_page = True
            except OSError as exc:
                errors.append(
                    ErrorItem(
                        message=f"markdown for this page could not be read: {exc}",
                        code="page_degraded",
                        page_no=page_no,
                    )
                )
                md_slots.append(
                    _FAILED_PAGE_MD.format(page_no=page_no, code="page_degraded")
                )
        else:
            md_slots.append(None)

        json_path = r.get("layout_info_path") if "json" in to_formats else None
        if json_path:
            try:
                with open(json_path, encoding="utf-8") as f:
                    json_slots.append(json.load(f))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    ErrorItem(
                        message=f"layout json for this page could not be read: {exc}",
                        code="page_degraded",
                        page_no=page_no,
                    )
                )
                json_slots.append(None)
        else:
            json_slots.append(_NO_JSON)

    # Markers alone are not content: a document where every page failed has no
    # markdown at all rather than a page of comments.
    md_content = (
        "\n\n---\n\n".join(s for s in md_slots if s is not None)
        if md_from_page
        else None
    )
    json_content = (
        [None if s is _NO_JSON else s for s in json_slots]
        if any(s is not _NO_JSON for s in json_slots)
        else None
    )
    return md_content, json_content, errors, pages_ok


_STEM_MAX_BYTES = 100


def _safe_stem(stem: str) -> str:
    """Build a filesystem-safe save_name for intermediate artifacts.

    The stem is only used to name temp files (``<stem>_page_<n>.json`` etc.),
    never the response filename, so truncating it is safe. Long non-ASCII names
    blow past the 255-byte NAME_MAX once encoded to UTF-8, so cap by bytes and
    append a digest to keep distinct sources distinct.
    """
    cleaned = re.sub(r"[/\\\x00]", "_", stem).strip() or "document"
    encoded = cleaned.encode("utf-8")
    if len(encoded) <= _STEM_MAX_BYTES:
        return cleaned
    digest = hashlib.sha1(encoded).hexdigest()[:8]
    truncated = encoded[: _STEM_MAX_BYTES - 9].decode("utf-8", errors="ignore").rstrip()
    return f"{truncated}_{digest}"


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
    stem = _safe_stem(Path(filename).stem)
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
            results = parser.parse_pdf(
                file_path, stem, prompt_mode, tmp_out,
                image_mode=image_mode, describe_script=describe_script,
                start_page=page_range[0] if page_range else 0,
                end_page=page_range[1] if page_range else None,
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
            errors.append(ErrorItem(
                message="Parser returned no results (0 renderable pages)",
                code="document_failed",
            ))
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename),
                status="failure",
                errors=errors,
                processing_time=elapsed,
            )

        md_content, json_content, page_errors, pages_ok = _assemble_outputs(
            results, to_formats
        )
        errors.extend(page_errors)
        text_content = (
            _extract_plain_text(md_content)
            if md_content and "text" in to_formats
            else None
        )

        # A fatal page error degrades the document; the HTTP status is decided
        # later, by enforce_failure_policy, once every source is done.
        fatal = [e for e in errors if e.code in FATAL_ERROR_CODES]
        if not fatal:
            doc_status = "success"
        elif pages_ok:
            doc_status = "partial_success"
        else:
            doc_status = "failure"

        elapsed = time.monotonic() - start
        logger.info(
            "convert %s: filename=%s pages=%d ok=%d fatal_errors=%d "
            "md_chars=%s json_pages=%s in %.2fs",
            doc_status, filename, len(results), pages_ok, len(fatal),
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
            status=doc_status,
            errors=errors,
            processing_time=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.exception(
            "convert error: filename=%s failed after %.2fs: %s",
            filename, elapsed, exc,
        )
        errors.append(ErrorItem(message=str(exc), code="document_failed"))
        return ConvertDocumentResponse(
            document=ExportDocumentResponse(filename=filename),
            status="failure",
            errors=errors,
            processing_time=elapsed,
        )
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


_CONTENT_TYPE_SUFFIX = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
}


async def _materialise_source(source: FileSourceRequest | HttpSourceRequest) -> tuple[str, str]:
    if isinstance(source, FileSourceRequest):
        suffix = Path(source.filename).suffix.lower()
        try:
            data = base64.b64decode(source.base64_string)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 content for '{source.filename}': {exc}",
            )
        if not data:
            raise HTTPException(
                status_code=400,
                detail=f"File '{source.filename}' is empty (0 bytes after base64 decode)",
            )
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        logger.debug(
            "materialised upload: filename=%s size=%dB -> %s",
            source.filename, len(data), path,
        )
        return path, source.filename

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(source.url, headers=source.headers)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download '{source.url}': {exc}",
        )

    # Take the filename from the URL path only — query strings would otherwise
    # corrupt the suffix and route PDFs to the image parser.
    filename = Path(urlparse(str(source.url)).path).name or "document.pdf"
    suffix = Path(filename).suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        suffix = _CONTENT_TYPE_SUFFIX.get(content_type, ".pdf")
        filename = f"{Path(filename).stem or 'document'}{suffix}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    logger.debug(
        "materialised url: %s -> filename=%s size=%dB -> %s",
        source.url, filename, len(resp.content), path,
    )
    return path, filename


def _fatal_errors(doc: ConvertDocumentResponse) -> list[ErrorItem]:
    return [e for e in doc.errors if e.code in FATAL_ERROR_CODES]


def _summarise_failure(doc: ConvertDocumentResponse) -> str:
    fatal = _fatal_errors(doc)
    if not fatal:
        return f"status={doc.status}"
    return f"{len(fatal)} error(s), first: {fatal[0].message}"


def enforce_failure_policy(
    results: list[ConvertDocumentResponse], allow_partial: bool
) -> None:
    """Turn failed conversions into a real HTTP error.

    Without ``allow_partial_results`` any document that is not fully successful
    fails the request; with it, only a request that recognised nothing at all
    does. Raises 502 when every fatal error came from the model backend, so a
    client can tell "the model is down" from "we broke"; 500 otherwise.
    """
    if not results:
        return

    if allow_partial:
        if any(r.status != "failure" for r in results):
            return
        offending = results
        message = f"all {len(results)} document(s) failed"
    else:
        offending = [r for r in results if r.status != "success"]
        if not offending:
            return
        message = f"{len(offending)} of {len(results)} document(s) failed"

    message += ": " + "; ".join(
        f"'{r.document.filename}': {_summarise_failure(r)}" for r in offending
    )
    codes = {e.code for r in offending for e in _fatal_errors(r)}
    status_code = 502 if codes == {"page_model_error"} else 500
    logger.warning("convert request failed (%d): %s", status_code, message)
    raise HTTPException(
        status_code=status_code,
        detail=ConvertFailureDetail(
            message=message,
            documents=[
                DocumentErrorSummary(
                    filename=r.document.filename,
                    status=r.status,
                    errors=r.errors,
                    processing_time=r.processing_time,
                )
                for r in results
            ],
        ).model_dump(),
    )


async def convert_source(
    parser: "DotsMOCRParser",
    request: ConvertDocumentsRequest,
    on_start: Optional[Callable[[], Awaitable[None]]] = None,
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
                # Fires once actual work begins, i.e. after a concurrency slot
                # is acquired — lets async tasks stay PENDING while queued.
                if on_start is not None:
                    await on_start()
                    on_start = None
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

    enforce_failure_policy(results, request.options.allow_partial_results)
    return results
