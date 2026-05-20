from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class TaskRecord:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    result: Optional[Any] = None
    error_message: Optional[str] = None


class TaskManager:
    def __init__(self, max_age_seconds: int = 3600) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()
        self._max_age = max_age_seconds

    async def create(self) -> str:
        task_id = str(uuid.uuid4())
        async with self._lock:
            self._tasks[task_id] = TaskRecord(task_id=task_id)
        return task_id

    async def set_running(self, task_id: str) -> None:
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.RUNNING

    async def set_success(self, task_id: str, result: Any) -> None:
        async with self._lock:
            if task_id in self._tasks:
                rec = self._tasks[task_id]
                rec.status = TaskStatus.SUCCESS
                rec.result = result

    async def set_failure(self, task_id: str, error: str) -> None:
        async with self._lock:
            if task_id in self._tasks:
                rec = self._tasks[task_id]
                rec.status = TaskStatus.FAILURE
                rec.error_message = error

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def position(self, task_id: str) -> int:
        async with self._lock:
            pending = [
                t for t in self._tasks.values()
                if t.status == TaskStatus.PENDING
            ]
            pending.sort(key=lambda t: t.created_at)
            for i, t in enumerate(pending):
                if t.task_id == task_id:
                    return i
            return 0

    async def gc(self) -> None:
        cutoff = time.time() - self._max_age
        async with self._lock:
            stale = [tid for tid, rec in self._tasks.items() if rec.created_at < cutoff]
            for tid in stale:
                del self._tasks[tid]
