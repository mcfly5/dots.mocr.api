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
| `VLLM_NUM_THREAD` | `64` | Concurrent page inference calls sent to vLLM **per document** |
| `VLLM_TIMEOUT` | `300` | Per-page inference timeout in seconds. Raise it if large pages queue behind high `VLLM_NUM_THREAD` on a slow GPU |
| `VLLM_FALLBACK_HOST` | _(unset)_ | When set, enables a fallback vLLM endpoint used when the main one fails (see [Fallback model](#fallback-model)) |
| `VLLM_FALLBACK_PORT` | `VLLM_PORT` | Fallback vLLM server port |
| `VLLM_FALLBACK_PROTOCOL` | `VLLM_PROTOCOL` | Fallback `http` or `https` |
| `VLLM_FALLBACK_MODEL_NAME` | `VLLM_MODEL_NAME` | Model name passed to the fallback server |
| `VLLM_FALLBACK_API_KEY` | `API_KEY` | API key for the fallback server |
| `VLLM_FALLBACK_COOLDOWN` | `0` | Seconds to route straight to the fallback after the main model becomes unreachable; `0` (default) retries the main model on every call |
| `VLLM_FALLBACK_STRIP_THINKING` | `1` | Strip `<think>…</think>` reasoning from the fallback's answers; set `0` to keep them verbatim |
| `VLLM_FALLBACK_PROMPTS` | `dots` | Prompts sent to the fallback: `dots` = the main model's prompts; `generic` = prompts written for a general-purpose VLM (layout and OCR modes) |
| `VLLM_FALLBACK_BBOX_SCALE` | `0` | Coordinate range of the fallback's layout boxes: `0` = pixels of the input image (dots.mocr, Qwen2.5-VL), `1000` = relative 0–1000 (Qwen3-VL / Qwen3.5) |
| `MOCR_MAX_CONCURRENT` | `2` | Max documents converted concurrently; extra requests queue and wait |
| `MOCR_API_KEY` | _(unset)_ | When set, enables API key auth on all `/v1` endpoints |
| `MOCR_OUTPUT_DIR` | `/tmp/mocr_output` | Base directory for temporary output files |
| `MOCR_TASK_TTL` | `3600` | Seconds to retain async task records in memory |
| `MOCR_LOG_LEVEL` | `info` | Log verbosity for both the API layer and the OCR pipeline. Set to `debug` to trace every stage (page render, vLLM request/response timing, layout parsing, markdown output) |

**Example with auth enabled:**
```bash
MOCR_API_KEY=mysecret VLLM_HOST=gpu-server python serve.py --port 8003
```

### Concurrency & GPU load

OCR is GPU-bound: the actual inference runs on the vLLM server. Two settings cap
how much work is in flight at once, so concurrent requests don't overwhelm the GPU
(which otherwise shows up as vLLM timeouts, OOM, or—because PyMuPDF rasterization
races under load—occasional `0 pages` failures):

- **`MOCR_MAX_CONCURRENT`** limits how many documents are processed at the same time.
  It's an async semaphore — requests beyond the limit **wait their turn** (they are
  not rejected), then run when a slot frees.
- **`VLLM_NUM_THREAD`** limits how many pages of a single document are sent to vLLM
  concurrently.

Effective concurrent inference ≈ `MOCR_MAX_CONCURRENT × VLLM_NUM_THREAD`. For a
single GPU, start with `MOCR_MAX_CONCURRENT=1` (or `2`) and tune `VLLM_NUM_THREAD`
to keep the GPU busy without triggering vLLM timeouts:

```bash
MOCR_MAX_CONCURRENT=1 VLLM_NUM_THREAD=32 python serve.py --port 8003
```

For large or bursty batches, prefer the asynchronous `/v1/convert/*/async`
endpoints (fire-and-forget + polling) over holding a synchronous connection open.

### Fallback model

Set `VLLM_FALLBACK_HOST` to add a second OpenAI-compatible endpoint. It should serve
dots.mocr or a model that follows the same prompts, because its answers are
post-processed the same way. A general Qwen-VL model also works for layout pages:
a ```` ```json ```` fence around the answer and Qwen grounding keys (`bbox_2d`,
`text_content`) are accepted, and a missing `category` defaults to `Text`, so such
pages get text but no headings, tables or formulas. For Qwen3-VL / Qwen3.5, set
`VLLM_FALLBACK_BBOX_SCALE=1000`, because they emit relative 0–1000 boxes.

For such a model, also set `VLLM_FALLBACK_PROMPTS=generic`. It then gets its own
layout and OCR prompts, which ask for a category and for tables as HTML and
formulas as LaTeX, so its pages keep their structure. Grounding, SVG and custom
prompts are sent unchanged.

```bash
VLLM_FALLBACK_HOST=gpu-b VLLM_FALLBACK_MODEL_NAME=Qwen3.5-4B \
VLLM_FALLBACK_PROMPTS=generic VLLM_FALLBACK_BBOX_SCALE=1000 python serve.py
```

- The fallback is used only when the main call fails with an upstream error
  (connection refused, timeout, 5xx, 429). A 4xx (e.g. prompt or image too long)
  is the request's fault and is **not** retried, and neither is an empty response.
- With `VLLM_FALLBACK_COOLDOWN` > 0, once the main model is unreachable (connection
  refused or timeout — a single 5xx does not count), calls go straight to the fallback for
  `VLLM_FALLBACK_COOLDOWN` seconds, so a stuck backend does not cost a full
  `VLLM_TIMEOUT` on every page. After that, the main model is tried again. If the
  fallback fails during the cooldown, the main model is probed at once and, if it
  answers, the cooldown ends early.
- Thinking models work as a fallback: inline `<think>…</think>` reasoning is stripped
  from their answers (`VLLM_FALLBACK_STRIP_THINKING`). A reasoning-only answer — e.g.
  cut off by `max_completion_tokens` mid-thought — fails the page with `page_failed`.
  The main model's output is never stripped.
- Pages handled by the fallback are still `success`. Each one is flagged with a
  non-fatal `page_fallback_model` entry in `errors`.
- If both models fail, the page fails with `page_model_error`, the same as without a fallback.

```bash
VLLM_HOST=gpu-a VLLM_FALLBACK_HOST=gpu-b VLLM_FALLBACK_PORT=8001 python serve.py --port 8003
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
- `image_mode`, `to_formats`, `allow_partial_results` _(optional)_ — plain form fields that override the same keys in `options_json`

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
| `describe_script` | string | `null` | Python script used when `image_mode` is `"describe"`. Must live in the repository's `scripts/` directory |
| `allow_partial_results` | bool | `false` | Return the pages that were recognised instead of failing the request when some pages could not be processed. See **Page Failures** below |

### Image Handling

The `image_mode` option controls how detected `Picture` layout cells are embedded in the `md_content` output.

| `image_mode` | Markdown output | Notes |
|---|---|---|
| `base64` _(default)_ | `![](data:image/png;base64,...)` | Crops the region and inlines it as a data URI. Self-contained but increases response size. |
| `file_ref` | `![](picture_0.png)` | Emits a filename placeholder — no image data is included. Useful when you handle images separately. |
| `describe` | `> [Image: <description>]` | Calls `describe_script` with the cropped image path and embeds its stdout as text. |
| `ocr` | the extracted text | Crops the region and runs it through OCR. The text replaces the image in `md_content` and also populates the cell's `text` field in the `json` output. Costs one extra model call per `Picture` cell. |

**`ocr` mode notes:**

The layout pass returns no text for `Picture` cells by design, so this mode issues
one additional inference per picture, cropping the region and asking the model to
extract its text. Use it for scanned figures, screenshots of tables, or stamped
blocks whose text would otherwise be lost.

Cost is one call per picture on top of the one call per page, and the calls are
sequential within a page — a page with many figures takes proportionally longer.
Pictures containing no text are dropped from the output entirely. If a call fails
the cell is skipped with a logged warning and the rest of the conversion proceeds.
Has no effect when `do_ocr` is `false` (detection-only mode produces no text).

**`describe` mode setup:**

Provide `describe_script` as the name of a Python script inside the repository's
`scripts/` directory (e.g. `describe_image.py` or `scripts/describe_image.py`;
paths outside `scripts/` are rejected with `422`). The server calls it as:
```
python <describe_script> <tmp_image_path>
```
The script should print a single-line description to stdout. If the script fails,
times out (30s), or prints nothing, the cell degrades to `> [Image]` and the rest
of the conversion proceeds. A stub is provided at `scripts/describe_image.py` —
replace its body with real logic (e.g. a VLM call).

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

**Example — ocr mode:**
```bash
curl -s -X POST http://localhost:8003/v1/convert/file \
  -F "files=@document.pdf" \
  -F 'options_json={"image_mode":"ocr","to_formats":["md","json"]}'
```

---

### Page Failures

Pages are processed independently, so one bad page no longer discards the rest of the
document. What happens next depends on `allow_partial_results`:

| | `false` (default) | `true` |
|---|---|---|
| All pages fine | `200`, `status: "success"` | same |
| Some pages failed | `500` / `502` | `200`, `status: "partial_success"`, recognised pages returned |
| No page produced output | `500` / `502` | `500` / `502` |
| Several sources, any failing | `500` / `502` | `200`, unless nothing at all was recognised |

`502` is returned when *every* fatal error came from the model backend (vLLM
unreachable, timed out, or answering 5xx/429) — i.e. the document is probably fine and the
request is worth retrying. `422` is returned when every failing document could not be
decoded at all (`document_invalid`). Anything else is `500`.

The error body carries the same per-page structure as a successful response:

```json
{
  "detail": {
    "message": "1 of 2 document(s) failed: 'scan.pdf': 2 error(s), first: Connection error.",
    "documents": [
      {
        "filename": "scan.pdf",
        "status": "failure",
        "errors": [
          {"code": "page_model_error", "page_no": 3, "message": "Connection error."},
          {"code": "page_skipped", "page_no": 7,
           "message": "Page contains an oversized embedded image (xref: 12, size: 9000x9000); ..."}
        ],
        "processing_time": 12.4
      }
    ]
  }
}
```

**Error codes** (`errors[].code`):

| Code | Fatal | Meaning |
|------|-------|---------|
| `page_failed` | yes | The page raised during processing, or the model answered but nothing usable could be recovered from it (not valid layout JSON and no salvageable text) |
| `page_model_error` | yes | The vLLM backend failed for this page (connection, timeout, upstream 5xx/429) |
| `page_skipped` | yes | The renderer refused the page (oversized embedded image, empty pixmap) |
| `page_degraded` | no | Content was recovered, but not cleanly — layout JSON did not parse and text was salvaged by the fallback cleaner, or a page artifact could not be read back |
| `page_empty_response` | no | The model returned an empty response for this page |
| `page_fallback_model` | no | The main model was unavailable; this page was processed by the fallback model |
| `document_failed` | yes | The whole document failed (0 renderable pages, internal error) |
| `document_invalid` | yes | The upload could not be decoded as an image or PDF |

Non-fatal codes are reported in `errors` but never change `status` or the HTTP status —
they exist so a blank or layout-less page is distinguishable from a genuinely blank one.

With `allow_partial_results: true`, failed pages **keep their slot** so page numbers stay
usable: `json_content[i]` is `null` for a failed page, and `md_content` carries a marker
comment between the `---` separators:

```
...page 0 markdown...

---

<!-- dots.mocr: page 1 failed (page_model_error) -->

---

...page 2 markdown...
```

`page_no` is the true 0-indexed PDF page number (the same numbering as `page_range`), so
a skipped page does not shift the pages that follow it. Markers are stripped from
`text_content`.

```bash
# get whatever was recognised, even if some pages failed
curl -X POST http://localhost:8003/v1/convert/file \
  -F "files=@scan.pdf" \
  -F "allow_partial_results=true"
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
     status:          "success" | "partial_success" | "failure"
     errors:          list of {message: string,
                               code: string|null,     # see Page Failures
                               page_no: int|null}     # 0-indexed PDF page
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
| `400` | Empty upload (0 bytes) or invalid base64 content |
| `401` | Missing or invalid `X-API-Key` (when auth is enabled) |
| `422` | Validation error: unsupported file extension, bad `prompt_mode`, malformed `page_range`, invalid `options_json`/`to_formats`, disallowed `describe_script`; or the file could not be decoded (`document_invalid`, see **Page Failures**) |
| `502` | HTTP source download failed, or every page failed because of the model backend (see **Page Failures**) |
| `404` | Task ID not found (async endpoints) |
| `202` | Task result requested but not yet complete |
| `500` | Conversion failed: pages could not be processed and `allow_partial_results` is not set, or an internal error |
| `200` with `"status":"partial_success"` | Some pages failed and `allow_partial_results` is set |

Failures of a conversion (`500`/`502`/`422`) carry a `detail` object with per-document,
per-page errors — see **Page Failures** for the shape and the code list. The async
endpoints replay the same status code and body from `GET /v1/result/{task_id}`.

> **Changed behaviour.** Earlier versions answered `200` with `"status":"failure"` for
> any parser error, and silently dropped pages the renderer refused. Both now produce a
> `500`/`502` unless `allow_partial_results` is set.

---

## Interactive API Docs

Once the server is running:

- **Swagger UI** — `http://localhost:8003/docs`
- **ReDoc** — `http://localhost:8003/redoc`
- **OpenAPI JSON** — `http://localhost:8003/openapi.json`
