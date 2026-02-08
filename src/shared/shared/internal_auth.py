"""
Internal Authentication System — THE HIVE
Secures inter-service communication using JWT tokens with a shared secret.
"""

import jwt
import time
import logging
from typing import Optional
from shared.config import get_settings

logger = logging.getLogger(__name__)

# Secret key for internal signing (Sync with all experts)
# In production, this should come from a secure env var or Vault.
INTERNAL_SECRET = "hive-swarm-distributed-secret-2026"

class InternalAuth:
    """
    Handles generation and validation of tokens for internal expert requests.
    """

    @staticmethod
    def generate_token(source_agent: str) -> str:
        """
        Generates a short-lived token for an internal request.
        """
        payload = {
            "iss": "hive-core",
            "sub": "internal-swarm-request",
            "src": source_agent,
            "iat": int(time.time()),
            "exp": int(time.time()) + 60  # 60 seconds expiry (high speed swarm)
        }
        return jwt.encode(payload, INTERNAL_SECRET, algorithm="HS256")

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """
        Verifies an internal token and returns the payload if valid.
        """
        try:
            payload = jwt.decode(token, INTERNAL_SECRET, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Internal token expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid internal token: {e}")
        return None

def get_internal_headers(agent_name: str) -> dict:
    """
    Helper to get the required headers for an internal request.
    """
    token = InternalAuth.generate_token(agent_name)
    return {
        "X-Hive-Internal-Token": token,
        "User-Agent": f"TheHive/{agent_name}"
    }
