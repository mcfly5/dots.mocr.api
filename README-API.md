# dots.mocr API Server

REST API server for dots.mocr, compatible with the [docling-serve](https://github.com/docling-project/docling-serve) interface. Clients built against docling-serve can point at this server with no code changes.

---

## Architecture

```
Client
  └─► dots.mocr API server  (serve.py, port 8003)
        └─► vLLM server     (port 8000, runs dots.mocr model)
```

The API server wraps `DotsMOCRParser`: it materialises document sources to temp files, calls the parser (which sends inference requests to vLLM), reads the outputs into memory, and returns structured JSON. Temp files are cleaned up after each request.

---

## Prerequisites

1. **vLLM server** running the dots.mocr model:
   ```bash
   CUDA_VISIBLE_DEVICES=0 vllm serve rednote-hilab/dots.mocr \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization 0.9 \
     --chat-template-content-format string \
     --trust-remote-code
   ```

  Local Nvidia RTX3060 12Gb with 40K context
  ```bash
  vllm serve rednote-hilab/dots.mocr --chat-template-content-format string --trust-remote-code --download-dir weights/ --gpu-memory-utilization 0.9 --max-model-len 40000 --enforce-eager
  ```
2. **API server dependencies** (in addition to the base `requirements.txt`):
   ```bash
   pip install fastapi "uvicorn[standard]" python-multipart httpx
   ```

---

## Starting the Server

```bash
python serve.py [--host HOST] [--port PORT] [--workers N] [--reload]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8003` | Listen port (8003 avoids conflict with vLLM on 8000) |
| `--workers` | `1` | Uvicorn worker processes. Keep at 1 to share the parser singleton |
| `--reload` | off | Hot-reload for development |

**Example:**
```bash
python serve.py --port 8003
```

---

## Configuration

All settings are via environment variables. None are required — defaults work for a local vLLM setup.

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST` | `localhost` | vLLM server hostname |
| `VLLM_PORT` | `8000` | vLLM server port |
| `VLLM_PROTOCOL` | `http` | `http` or `https` |
| `VLLM_MODEL_NAME` | `model` | Model name passed to vLLM |
| `MOCR_API_KEY` | _(unset)_ | When set, enables API key auth on all `/v1` endpoints |
| `MOCR_OUTPUT_DIR` | `/tmp/mocr_output` | Base directory for temporary output files |
| `MOCR_TASK_TTL` | `3600` | Seconds to retain async task records in memory |

**Example with auth enabled:**
```bash
MOCR_API_KEY=mysecret VLLM_HOST=gpu-server python serve.py --port 8003
```

---

## Authentication

Authentication is **disabled by default**. Set `MOCR_API_KEY` to enable it.

When enabled, all `/v1/*` endpoints require the header:
```
X-API-Key: <your-key>
```

Health and version endpoints (`/health`, `/ready`, `/version`) are always public.

---

## API Endpoints

### Health & Status

#### `GET /health`
Always returns 200.
```json
{"status": "ok"}
```

#### `GET /ready` · `/readyz` · `/livez`
Returns 200 when the parser is initialised, 503 otherwise.
```json
{"status": "ready"}
```

#### `GET /version`
```json
{"name": "dots.mocr-serve", "version": "1.0.0", "docling_serve_compat": "0.1"}
```

---

### Document Conversion — Synchronous

Both endpoints return results immediately after processing.

#### `POST /v1/convert/source`

Convert documents from URLs or base64-encoded file content.

**Request body** (`application/json`):
```json
{
  "sources": [
    {
      "kind": "file",
      "base64_string": "<base64-encoded file bytes>",
      "filename": "document.pdf"
    }
  ],
  "options": {
    "do_ocr": true,
    "to_formats": ["md", "json"],
    "page_range": null,
    "prompt_mode": null
  }
}
```

Or from a URL:
```json
{
  "sources": [
    {
      "kind": "http",
      "url": "https://example.com/document.pdf",
      "headers": {}
    }
  ]
}
```

**Response** (`application/json`):
```json
[
  {
    "document": {
      "filename": "document.pdf",
      "md_content": "# Title\n\nParagraph text...",
      "json_content": [
        [
          {"bbox": [x1, y1, x2, y2], "category": "text", "text": "..."},
          {"bbox": [x1, y1, x2, y2], "category": "title", "text": "..."}
        ]
      ],
      "text_content": null,
      "html_content": null,
      "doctags_content": null
    },
    "status": "success",
    "errors": [],
    "processing_time": 3.14
  }
]
```

`json_content` is a list of pages; each page is a list of layout cells with bounding boxes.

---

#### `POST /v1/convert/file`

Convert documents via multipart file upload.

**Request** (`multipart/form-data`):
- `files` — one or more files (`.jpg`, `.jpeg`, `.png`, `.pdf`)
- `options_json` _(optional)_ — JSON string of `ConvertDocumentsOptions` (default: `{}`)

**curl example:**
```bash
curl -X POST http://localhost:8003/v1/convert/file \
  -F "files=@document.pdf" \
  -F 'options_json={"to_formats":["md"]}'
```

Multiple files:
```bash
curl -X POST http://localhost:8003/v1/convert/file \
  -F "files=@page1.jpg" \
  -F "files=@page2.jpg"
```

**Response:** same as `/v1/convert/source`.

---

### Document Conversion — Asynchronous

Submit a job and poll for completion. Useful for large PDFs or batch workloads.

#### `POST /v1/convert/source/async`

Same request body as `/v1/convert/source`. Returns `202 Accepted` immediately.

```json
{"task_id": "3f2a1b4c-..."}
```

#### `POST /v1/convert/file/async`

Same as `/v1/convert/file` (multipart), returns `202 Accepted` with `task_id`.

---

### Task Management

#### `GET /v1/status/poll/{task_id}`

Poll task status.

```json
{
  "task_id": "3f2a1b4c-...",
  "task_status": "running",
  "task_position": null,
  "error_message": null
}
```

`task_status` values: `pending` · `running` · `success` · `failure`

`task_position` is set (integer queue position) only while status is `pending`.

#### `GET /v1/result/{task_id}`

Fetch the result once `task_status` is `success`. Returns the same response shape as the sync endpoints.

- `404` — task not found (or expired)
- `202` — task not yet complete
- `500` — task failed (detail contains error message)

---

## Request Options Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `do_ocr` | bool | `true` | Extract text. `false` returns layout bounding boxes only |
| `to_formats` | list | `["md","json"]` | Output formats: `"md"`, `"json"`, `"text"` |
| `page_range` | `[start, end]` | `null` | 0-indexed, inclusive. PDF only. E.g. `[0, 4]` for first 5 pages |
| `prompt_mode` | string | `null` | Override the prompt used. See table below |
| `image_mode` | string | `"base64"` | How `Picture` cells are rendered in Markdown output. See **Image Handling** below |
| `describe_script` | string | `null` | Path to a Python script used when `image_mode` is `"describe"` |

### Image Handling

The `image_mode` option controls how detected `Picture` layout cells are embedded in the `md_content` output.

| `image_mode` | Markdown output | Notes |
|---|---|---|
| `base64` _(default)_ | `![](data:image/png;base64,...)` | Crops the region and inlines it as a data URI. Self-contained but increases response size. |
| `file_ref` | `![](picture_0.png)` | Emits a filename placeholder — no image data is included. Useful when you handle images separately. |
| `describe` | `> [Image: <description>]` | Calls `describe_script` with the cropped image path and embeds its stdout as text. |

**`describe` mode setup:**

Provide `describe_script` as a path to a Python script. The server calls it as:
```
python <describe_script> <tmp_image_path>
```
The script should print a single-line description to stdout. A stub is provided at `scripts/describe_image.py` — replace its body with real logic (e.g. a VLM call).

**Example — file_ref mode:**
```bash
curl -s -X POST http://localhost:8003/v1/convert/file \
  -F "files=@document.pdf" \
  -F 'options_json={"image_mode":"file_ref","to_formats":["md"]}'
```

**Example — describe mode:**
```bash
curl -s -X POST http://localhost:8003/v1/convert/file \
  -F "files=@document.pdf" \
  -F 'options_json={"image_mode":"describe","describe_script":"scripts/describe_image.py","to_formats":["md"]}'
```

---

### Prompt Modes

| `prompt_mode` | Description |
|--------------|-------------|
| `prompt_layout_all_en` | Full layout detection + OCR → structured JSON + Markdown **(default)** |
| `prompt_layout_only_en` | Layout bounding boxes only, no text extraction |
| `prompt_ocr` | Plain text extraction without layout |
| `prompt_web_parsing` | Webpage layout parsing |
| `prompt_scene_spotting` | Scene text detection and recognition |
| `prompt_image_to_svg` | Generate SVG code reconstructing the image |
| `prompt_general` | Free-form question answering (use with a custom prompt) |

When `prompt_mode` is not set, the server selects automatically: `do_ocr=true` → `prompt_layout_all_en`, `do_ocr=false` → `prompt_layout_only_en`.

---

## Supported File Types

`.jpg` · `.jpeg` · `.png` · `.pdf`

Other extensions are rejected with `422 Unprocessable Entity`.

---

## Response Structure

```
list[ConvertDocumentResponse]
  └─ document: ExportDocumentResponse
       ├─ filename:        original filename
       ├─ md_content:      Markdown text (null if not requested)
       ├─ json_content:    list of pages, each page is a list of layout cells
       │                   cell: {bbox, category, text, ...}
       ├─ text_content:    plain text stripped of Markdown (null if not requested)
       ├─ html_content:    always null (not supported)
       └─ doctags_content: always null (not supported)
     status:          "success" | "failure" | "partial_success"
     errors:          list of {message: string}
     processing_time: seconds (float)
```

---

## Examples

### Convert a local image (bash)

```bash
# Encode file and call sync endpoint
B64=$(base64 -w0 demo/demo_image1.jpg)

curl -s -X POST http://localhost:8003/v1/convert/source \
  -H "Content-Type: application/json" \
  -d "{\"sources\":[{\"kind\":\"file\",\"base64_string\":\"$B64\",\"filename\":\"demo_image1.jpg\"}]}" \
  | python3 -m json.tool
```

### Convert a local image (multipart)

```bash
curl -s -X POST http://localhost:8003/v1/convert/file \
  -F "files=@demo/demo_image1.jpg" \
  | python3 -m json.tool
```

### Convert a PDF, first 3 pages, Markdown only

```bash
curl -s -X POST http://localhost:8003/v1/convert/file \
  -F "files=@demo/demo_pdf.pdf" \
  -F 'options_json={"to_formats":["md"],"page_range":[0,2]}' \
  | python3 -m json.tool
```

### Async conversion + polling (bash)

```bash
# Submit
TASK=$(curl -s -X POST http://localhost:8003/v1/convert/file/async \
  -F "files=@demo/demo_pdf.pdf" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

echo "Task: $TASK"

# Poll until done
while true; do
  STATUS=$(curl -s "http://localhost:8003/v1/status/poll/$TASK" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['task_status'])")
  echo "Status: $STATUS"
  [ "$STATUS" = "success" ] || [ "$STATUS" = "failure" ] && break
  sleep 2
done

# Fetch result
curl -s "http://localhost:8003/v1/result/$TASK" | python3 -m json.tool
```

### With authentication

```bash
# Start server with auth
MOCR_API_KEY=mysecret python serve.py

# Call with header
curl -s -X POST http://localhost:8003/v1/convert/file \
  -H "X-API-Key: mysecret" \
  -F "files=@demo/demo_image1.jpg"
```

### Python client

```python
import base64
import httpx

def convert_file(path: str, server: str = "http://localhost:8003") -> dict:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = httpx.post(
        f"{server}/v1/convert/source",
        json={
            "sources": [{"kind": "file", "base64_string": b64, "filename": path}],
            "options": {"to_formats": ["md", "json"]},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

results = convert_file("demo/demo_image1.jpg")
print(results[0]["document"]["md_content"])
```

---

## Error Reference

| HTTP Status | Cause |
|-------------|-------|
| `401` | Missing or invalid `X-API-Key` (when auth is enabled) |
| `422` | Validation error: unsupported file extension, bad `prompt_mode`, malformed `page_range` |
| `502` | HTTP source download failed |
| `404` | Task ID not found (async endpoints) |
| `202` | Task result requested but not yet complete |
| `500` | Async task failed internally |
| `200` with `"status":"failure"` | Parser error during conversion (docling-serve convention) |

---

## Interactive API Docs

Once the server is running:

- **Swagger UI** — `http://localhost:8003/docs`
- **ReDoc** — `http://localhost:8003/redoc`
- **OpenAPI JSON** — `http://localhost:8003/openapi.json`
