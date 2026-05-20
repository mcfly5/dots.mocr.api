#!/usr/bin/env python3
"""
dots.mocr API server — docling-serve-compatible REST interface.

Usage:
    python serve.py [--host HOST] [--port PORT] [--workers N] [--reload]

Environment variables (all optional):
    VLLM_HOST         vLLM server hostname     (default: localhost)
    VLLM_PORT         vLLM server port         (default: 8000)
    VLLM_PROTOCOL     http | https             (default: http)
    VLLM_MODEL_NAME   model name               (default: model)
    MOCR_API_KEY      enables X-API-Key auth when set
    MOCR_OUTPUT_DIR   base dir for temp output (default: /tmp/mocr_output)
    MOCR_TASK_TTL     async task TTL in secs   (default: 3600)
"""
import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="dots.mocr API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8003, help="Listen port (default: 8003)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Uvicorn worker processes. Keep at 1 to share the parser singleton.",
    )
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev only)")
    args = parser.parse_args()

    uvicorn.run(
        "dots_mocr.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
