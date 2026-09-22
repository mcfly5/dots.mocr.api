import os

import pytest
from fastapi.testclient import TestClient

from dots_mocr.api import app as app_mod
from dots_mocr.api.task_manager import TaskManager
from dots_mocr.parser import DotsMOCRParser


class _StubParser:
    """parse_pdf returns one good page and one failed page."""

    def parse_pdf(self, input_path, filename, prompt_mode, save_dir, **kwargs):
        md_path = os.path.join(save_dir, "p0.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("page zero")
        return [
            {"page_no": 0, "md_content_path": md_path, "file_path": input_path},
            {"page_no": 1, "file_path": input_path,
             "error": {"code": "page_failed", "message": "boom"}},
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_mod, "_parser", _StubParser())
    monkeypatch.setattr(app_mod, "_task_manager", TaskManager())
    # No `with`: the lifespan would replace the stubs with a real parser.
    return TaskClient(app_mod.app)


class TaskClient(TestClient):
    def upload(self, path, name, data, **form):
        return self.post(path, files={"files": (name, data)}, data=form)


_PDF = b"%PDF-1.4 stub"


def test_partial_failure_is_500_by_default(client):
    resp = client.upload("/v1/convert/file", "doc.pdf", _PDF)
    assert resp.status_code == 500
    doc = resp.json()["detail"]["documents"][0]
    assert doc["status"] == "partial_success"
    assert doc["errors"][0]["code"] == "page_failed"


def test_partial_failure_allowed(client):
    resp = client.upload(
        "/v1/convert/file", "doc.pdf", _PDF, allow_partial_results="true",
        to_formats="md,text",
    )
    assert resp.status_code == 200, resp.text
    doc = resp.json()[0]
    assert doc["status"] == "partial_success"
    assert doc["document"]["md_content"].startswith("page zero")
    assert doc["document"]["text_content"].startswith("page zero")
    assert "failed" not in doc["document"]["text_content"]


def test_async_result_replays_sync_failure(client):
    sync = client.upload("/v1/convert/file", "doc.pdf", _PDF)
    task = client.upload("/v1/convert/file/async", "doc.pdf", _PDF)
    assert task.status_code == 202
    result = client.get(f"/v1/result/{task.json()['task_id']}")
    assert result.status_code == sync.status_code == 500
    sync_detail, async_detail = sync.json()["detail"], result.json()["detail"]
    for d in sync_detail["documents"] + async_detail["documents"]:
        d.pop("processing_time")
    assert async_detail == sync_detail


def test_undecodable_image_is_422(monkeypatch, client):
    # The real parser: decoding fails before any inference is attempted.
    monkeypatch.setattr(app_mod, "_parser", DotsMOCRParser(ip="unused"))
    resp = client.upload("/v1/convert/file", "scan.png", b"definitely not a png")
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["documents"][0]["errors"][0]["code"] == "document_invalid"
