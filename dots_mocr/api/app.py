from __future__ import annotations

import asyncio
import base64
import os
from contextlib import asynccontextmanager
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

from dots_mocr.api.auth import get_api_key_dependency
from dots_mocr.api.datamodel.requests import (
    ConvertDocumentsOptions,
    ConvertDocumentsRequest,
    FileSourceRequest,
)
from dots_mocr.api.datamodel.responses import (
    ConvertDocumentResponse,
    TaskStatusResponse,
    VersionResponse,
)
from dots_mocr.api.engine import convert_source
from dots_mocr.api.task_manager import TaskManager, TaskStatus
from dots_mocr.parser import DotsMOCRParser

_parser: Optional[DotsMOCRParser] = None
_task_manager: Optional[TaskManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _parser, _task_manager
    _parser = DotsMOCRParser(
        protocol=os.environ.get("VLLM_PROTOCOL", "http"),
        ip=os.environ.get("VLLM_HOST", "localhost"),
        port=int(os.environ.get("VLLM_PORT", "8000")),
        model_name=os.environ.get("VLLM_MODEL_NAME", "model"),
        output_dir=os.environ.get("MOCR_OUTPUT_DIR", "/tmp/mocr_output"),
    )
    _task_manager = TaskManager(
        max_age_seconds=int(os.environ.get("MOCR_TASK_TTL", "3600"))
    )
    gc_task = asyncio.create_task(_gc_loop(_task_manager))
    yield
    gc_task.cancel()
    _parser = None
    _task_manager = None


async def _gc_loop(tm: TaskManager) -> None:
    while True:
        await asyncio.sleep(600)
        await tm.gc()


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

@app.post(
    "/v1/convert/file",
    response_model=list[ConvertDocumentResponse],
    dependencies=[Depends(_require_auth)],
)
async def convert_file_sync(
    files: list[UploadFile] = File(...),
    options_json: str = Form(default="{}"),
    parser: DotsMOCRParser = Depends(_get_parser),
):
    options = ConvertDocumentsOptions.model_validate_json(options_json)
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
    files: list[UploadFile] = File(...),
    options_json: str = Form(default="{}"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    parser: DotsMOCRParser = Depends(_get_parser),
    tm: TaskManager = Depends(_get_task_manager),
):
    options = ConvertDocumentsOptions.model_validate_json(options_json)
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
            status_code=500, detail=record.error_message or "Task failed"
        )
    return record.result


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _uploads_to_sources(files: list[UploadFile]) -> list[FileSourceRequest]:
    sources = []
    for upload in files:
        data = await upload.read()
        sources.append(
            FileSourceRequest(
                base64_string=base64.b64encode(data).decode(),
                filename=upload.filename or "upload.pdf",
            )
        )
    return sources


async def _run_async_task(
    tm: TaskManager,
    parser: DotsMOCRParser,
    request: ConvertDocumentsRequest,
    task_id: str,
) -> None:
    await tm.set_running(task_id)
    try:
        result = await convert_source(parser, request)
        await tm.set_success(task_id, result)
    except Exception as exc:
        await tm.set_failure(task_id, str(exc))
