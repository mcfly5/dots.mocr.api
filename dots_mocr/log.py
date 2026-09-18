"""Central loguru logger for the dots.mocr processing engine.

Every engine module imports the configured ``logger`` from here so that log
routing and verbosity live in one place. Set ``MOCR_LOG_LEVEL`` (e.g. INFO,
DEBUG) to control how much stage detail is emitted; defaults to INFO.

The API layer (``dots_mocr.api.*``) deliberately stays on the stdlib
``uvicorn.error`` logger — ``serve.py`` reads the same ``MOCR_LOG_LEVEL`` and
passes it to uvicorn, so one variable drives both.

This module imports only the standard library and loguru so it can be imported
from anywhere in the engine without risking circular imports.
"""

import os
import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("MOCR_LOG_LEVEL", "INFO").upper(),
    backtrace=False,
    # Keep tracebacks from dumping huge PIL image / numpy array locals.
    diagnose=False,
)

__all__ = ["logger"]
