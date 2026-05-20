from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key_dependency():
    expected_key = os.environ.get("MOCR_API_KEY", "")

    async def verify(api_key: str = Security(_API_KEY_HEADER)) -> None:
        if not expected_key:
            return
        if api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    return verify
