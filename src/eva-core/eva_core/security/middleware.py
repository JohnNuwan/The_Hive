"""
Core Authentication Middleware — THE HIVE
Replaces InternalAuthMiddleware to support both Internal and User authentication.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from shared.internal_auth import InternalAuth
from eva_core.services.auth import get_auth_service
import logging

logger = logging.getLogger(__name__)

class CoreAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that checks for EITHER a valid X-Hive-Internal-Token OR a valid Bearer Token.
    """

    def __init__(self, app, exclude_paths: list[str] = None):
        super().__init__(app)
        # Default excluded paths including login endpoint
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/auth/login"]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        for p in self.exclude_paths:
            if request.url.path == p or request.url.path.startswith(p + "/"):
                return await call_next(request)

        # 1. Check Internal Auth
        internal_token = request.headers.get("X-Hive-Internal-Token")
        if internal_token:
            payload = InternalAuth.verify_token(internal_token)
            if payload:
                request.state.source_agent = payload.get("src")
                request.state.is_internal = True
                request.state.user = None  # Internal agents are not users
                return await call_next(request)
            else:
                logger.warning(f"🚨 Invalid internal token from {request.client.host}")
                # Fall through to check User Auth

        # 2. Check User Auth (Bearer)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                # Ensure AuthService is available
                auth_service = get_auth_service()
                payload = auth_service.validate_token(token)
                if payload:
                    request.state.user = payload
                    request.state.is_internal = False
                    request.state.source_agent = None
                    return await call_next(request)
                else:
                    logger.warning(f"🚨 Invalid user token from {request.client.host}")
            except RuntimeError:
                # AuthService might not be initialized during tests or startup
                logger.error("AuthService not initialized")
            except Exception as e:
                logger.error(f"Error validating user token: {e}")

        # If neither is valid
        client_host = request.client.host if request.client else "unknown"
        logger.warning(f"🚨 Unauthorized access attempt from {client_host} to {request.url.path}")
        return JSONResponse(status_code=401, content={"detail": "Authentication required (Internal or User)"})
