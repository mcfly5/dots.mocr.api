from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import ValidationError

from dots_mocr.api.auth import get_api_key_dependency
from dots_mocr.api.datamodel.requests import (
    IMAGE_MODES,
    ConvertDocumentsOptions,
    ConvertDocumentsRequest,
    FileSourceRequest,
)
from dots_mocr.api.datamodel.responses import (
    ConvertDocumentResponse,
    TaskStatusResponse,
    VersionResponse,
)
logger = logging.getLogger("uvicorn.error")

from dots_mocr.api.engine import convert_source
from dots_mocr.api.task_manager import TaskManager, TaskStatus
from dots_mocr.parser import DotsMOCRParser

_parser: Optional[DotsMOCRParser] = None
_task_manager: Optional[TaskManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _parser, _task_manager
    protocol = os.environ.get("VLLM_PROTOCOL", "http")
    port = int(os.environ.get("VLLM_PORT", "8000"))
    model_name = os.environ.get("VLLM_MODEL_NAME", "model")
    _parser = DotsMOCRParser(
        protocol=protocol,
        ip=os.environ.get("VLLM_HOST", "localhost"),
        port=port,
        model_name=model_name,
        output_dir=os.environ.get("MOCR_OUTPUT_DIR", "/tmp/mocr_output"),
        num_thread=int(os.environ.get("VLLM_NUM_THREAD", "64")),
        fallback=_fallback_from_env(protocol, port, model_name),
        fallback_cooldown=float(os.environ.get("VLLM_FALLBACK_COOLDOWN", "0")),
    )
    _task_manager = TaskManager(
        max_age_seconds=int(os.environ.get("MOCR_TASK_TTL", "3600"))
    )
    gc_task = asyncio.create_task(_gc_loop(_task_manager))
    yield
    gc_task.cancel()
    _parser = None
    _task_manager = None


def _fallback_from_env(protocol: str, port: int, model_name: str) -> Optional[dict]:
    """Fallback vLLM endpoint config; enabled only when VLLM_FALLBACK_HOST is set."""
    host = os.environ.get("VLLM_FALLBACK_HOST")
    if not host:
        return None
    prompts = os.environ.get("VLLM_FALLBACK_PROMPTS", "dots")
    if prompts not in ("dots", "generic"):
        raise ValueError(f"VLLM_FALLBACK_PROMPTS must be 'dots' or 'generic', got {prompts!r}")
    return {
        "protocol": os.environ.get("VLLM_FALLBACK_PROTOCOL", protocol),
        "ip": host,
        "port": int(os.environ.get("VLLM_FALLBACK_PORT", str(port))),
        "model_name": os.environ.get("VLLM_FALLBACK_MODEL_NAME", model_name),
        "api_key": os.environ.get("VLLM_FALLBACK_API_KEY"),
        # Strip <think> reasoning from the fallback's answers (thinking models).
        "strip_thinking": os.environ.get("VLLM_FALLBACK_STRIP_THINKING", "1") != "0",
        # Range of the fallback's relative bbox coordinates (1000 for Qwen3-VL
        # style models); 0 means pixels of the input image, as dots.mocr emits.
        "bbox_scale": float(os.environ.get("VLLM_FALLBACK_BBOX_SCALE", "0")),
        # "dots": same prompts as the main model; "generic": prompts written for
        # a general-purpose VLM (see dict_promptmode_to_fallback_prompt).
        "prompts": prompts,
    }


async def _gc_loop(tm: TaskManager) -> None:
    while True:
        await asyncio.sleep(600)
        try:
            await tm.gc()
        except Exception:
            logger.exception("task GC failed; will retry next cycle")


app = FastAPI(
    title="dots.mocr serve",
    description="docling-serve-compatible REST API for dots.mocr OCR",
    version="1.0.0",
    lifespan=lifespan,
)

_require_auth = get_api_key_dependency()


def _get_parser() -> DotsMOCRParser:
    if _parser is None:
        raise HTTPException(status_code=503, detail="Parser not initialised")
    return _parser


def _get_task_manager() -> TaskManager:
    if _task_manager is None:
        raise HTTPException(status_code=503, detail="Task manager not initialised")
    return _task_manager


# ── Health / status ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
@app.get("/readyz")
@app.get("/livez")
async def ready(parser: DotsMOCRParser = Depends(_get_parser)):
    return {"status": "ready"}


@app.get("/version", response_model=VersionResponse)
async def version():
    return VersionResponse()


# ── Sync: JSON body sources ────────────────────────────────────────────────────

@app.post(
    "/v1/convert/source",
    response_model=list[ConvertDocumentResponse],
    dependencies=[Depends(_require_auth)],
)
async def convert_source_sync(
    request: ConvertDocumentsRequest,
    parser: DotsMOCRParser = Depends(_get_parser),
):
    return await convert_source(parser, request)


# ── Sync: multipart upload ─────────────────────────────────────────────────────

def _parse_form_options(
    options_json: str,
    image_mode: Optional[str] = None,
    to_formats: Optional[str] = None,
    allow_partial_results: Optional[bool] = None,
) -> ConvertDocumentsOptions:
    try:
        options = ConvertDocumentsOptions.model_validate_json(options_json)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid options_json: {exc}",
        )
    if image_mode is not None:
        if image_mode not in IMAGE_MODES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown image_mode '{image_mode}'. Valid: {sorted(IMAGE_MODES)}",
            )
        options.image_mode = image_mode
    if to_formats is not None:
        try:
            parsed = json.loads(to_formats)
        except (ValueError, TypeError):
            parsed = [f.strip() for f in to_formats.split(",") if f.strip()]
        # Rebuild so field validators run on the overridden value.
        data = options.model_dump()
        data["to_formats"] = parsed
        try:
            options = ConvertDocumentsOptions.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid to_formats: {exc}",
            )
    if allow_partial_results is not None:
        options.allow_partial_results = allow_partial_results
    return options


@app.post(
    "/v1/convert/file",
    response_model=list[ConvertDocumentResponse],
    dependencies=[Depends(_require_auth)],
)
async def convert_file_sync(
    files: list[UploadFile] = File(...),
    options_json: str = Form(default="{}"),
    image_mode: Optional[str] = Form(default=None),
    to_formats: Optional[str] = Form(default=None),
    allow_partial_results: Optional[bool] = Form(default=None),
    parser: DotsMOCRParser = Depends(_get_parser),
):
    options = _parse_form_options(
        options_json, image_mode, to_formats, allow_partial_results
    )
    logger.info("convert_file options: %s", options.model_dump())
    sources = await _uploads_to_sources(files)
    request = ConvertDocumentsRequest(sources=sources, options=options)
    return await convert_source(parser, request)


# ── Async: JSON body sources ───────────────────────────────────────────────────

@app.post(
    "/v1/convert/source/async",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_require_auth)],
)
async def convert_source_async(
    request: ConvertDocumentsRequest,
    background_tasks: BackgroundTasks,
    parser: DotsMOCRParser = Depends(_get_parser),
    tm: TaskManager = Depends(_get_task_manager),
):
    task_id = await tm.create()
    background_tasks.add_task(_run_async_task, tm, parser, request, task_id)
    return {"task_id": task_id}


# ── Async: multipart upload ────────────────────────────────────────────────────

@app.post(
    "/v1/convert/file/async",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_require_auth)],
)
async def convert_file_async(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    options_json: str = Form(default="{}"),
    parser: DotsMOCRParser = Depends(_get_parser),
    tm: TaskManager = Depends(_get_task_manager),
):
    options = _parse_form_options(options_json)
    # Read file bytes before returning — UploadFile is invalid after response is sent
    sources = await _uploads_to_sources(files)
    request = ConvertDocumentsRequest(sources=sources, options=options)
    task_id = await tm.create()
    background_tasks.add_task(_run_async_task, tm, parser, request, task_id)
    return {"task_id": task_id}


# ── Task polling ───────────────────────────────────────────────────────────────

@app.get(
    "/v1/status/poll/{task_id}",
    response_model=TaskStatusResponse,
    dependencies=[Depends(_require_auth)],
)
async def poll_task(
    task_id: str,
    tm: TaskManager = Depends(_get_task_manager),
):
    record = await tm.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    position = (
        await tm.position(task_id) if record.status == TaskStatus.PENDING else None
    )
    return TaskStatusResponse(
        task_id=task_id,
        task_status=record.status,
        task_position=position,
        error_message=record.error_message,
    )


@app.get(
    "/v1/result/{task_id}",
    response_model=list[ConvertDocumentResponse],
    dependencies=[Depends(_require_auth)],
)
async def get_result(
    task_id: str,
    tm: TaskManager = Depends(_get_task_manager),
):
    record = await tm.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if record.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        raise HTTPException(status_code=202, detail="Task not yet complete")
    if record.status == TaskStatus.FAILURE:
        raise HTTPException(
            status_code=record.error_status or 500,
            detail=(
                record.error_detail
                if record.error_detail is not None
                else record.error_message or "Task failed"
            ),
        )
    return record.result


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _uploads_to_sources(files: list[UploadFile]) -> list[FileSourceRequest]:
    sources = []
    for upload in files:
        filename = upload.filename or "upload.pdf"
        # Defensive: ensure we read from the start of the spooled part.
        try:
            await upload.seek(0)
        except (OSError, AttributeError):
            pass
        data = await upload.read()
        logger.info("received upload: filename=%s size=%dB", filename, len(data))
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Uploaded file '{filename}' is empty (0 bytes). The file "
                    "content did not reach the server — check the client upload "
                    "and any proxy/gateway in front of the API."
                ),
            )
        try:
            sources.append(
                FileSourceRequest(
                    base64_string=base64.b64encode(data).decode(),
                    filename=filename,
                )
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid upload '{filename}': {exc}",
            )
    return sources


def _detail_message(detail) -> str:
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail)


async def _run_async_task(
    tm: TaskManager,
    parser: DotsMOCRParser,
    request: ConvertDocumentsRequest,
    task_id: str,
) -> None:
    try:
        result = await convert_source(
            parser, request, on_start=partial(tm.set_running, task_id)
        )
        await tm.set_success(task_id, result)
    except HTTPException as exc:
        # Conversion failures (and download/validation errors) keep their status
        # code and structured body, so /v1/result answers like the sync route.
        await tm.set_failure(
            task_id,
            _detail_message(exc.detail),
            status_code=exc.status_code,
            detail=exc.detail,
        )
    except Exception as exc:
        await tm.set_failure(task_id, str(exc))
