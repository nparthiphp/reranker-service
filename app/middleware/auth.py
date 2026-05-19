"""app/middleware/auth.py — API key authentication. Optional."""
from __future__ import annotations
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.settings import get_settings

settings   = get_settings()
api_key_hdr = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_hdr)) -> str:
    """
    Verify API key if configured.
    If API_KEY is not set in .env → auth disabled (open access).
    """
    if not settings.api_key:
        # Auth disabled — allow all requests
        return "no-auth"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
