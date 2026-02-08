"""
Internal Auth Middleware — THE HIVE
FastAPI middleware to protect expert endpoints.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from shared.internal_auth import InternalAuth
import logging

logger = logging.getLogger(__name__)

class InternalAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that checks for a valid X-Hive-Internal-Token header.
    Allows opting out for health checks or public endpoints.
    """
    
    def __init__(self, app, exclude_paths: list[str] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        token = request.headers.get("X-Hive-Internal-Token")
        if not token:
            logger.warning(f"🚨 Unauthorized access attempt from {request.client.host} to {request.url.path}")
            raise HTTPException(status_code=401, detail="X-Hive-Internal-Token header missing")

        payload = InternalAuth.verify_token(token)
        if not payload:
            logger.error(f"🚨 Invalid or expired token from {request.client.host}")
            raise HTTPException(status_code=403, detail="Invalid or expired internal token")

        # Optional: Inject source agent into request state
        request.state.source_agent = payload.get("src")
        
        return await call_next(request)
