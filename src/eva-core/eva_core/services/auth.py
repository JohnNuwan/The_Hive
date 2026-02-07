"""
Authentication & Authorization Service — THE HIVE
JWT-based auth with Role-Based Access Control (RBAC).
Users stored in Redis. No external dependencies beyond stdlib + FastAPI.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# JWT — Implémentation stdlib-only (aucune dépendance PyJWT)
# ═══════════════════════════════════════════════════════════════════════════════


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def create_jwt_token(payload: dict, secret: str) -> str:
    """Crée un JWT HS256 signé."""
    header = {"alg": "HS256", "typ": "JWT"}
    h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    message = f"{h_b64}.{p_b64}"
    sig = hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
    return f"{message}.{_b64url_encode(sig)}"


def decode_jwt_token(token: str, secret: str) -> Optional[dict]:
    """Décode et vérifie un JWT HS256. Retourne None si invalide/expiré."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h_b64, p_b64, sig_b64 = parts
        message = f"{h_b64}.{p_b64}"
        expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
        actual = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected, actual):
            return None
        payload = json.loads(_b64url_decode(p_b64))
        if "exp" in payload and payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PASSWORD HASHING — stdlib PBKDF2-SHA256
# ═══════════════════════════════════════════════════════════════════════════════


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return hashed.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return hmac.compare_digest(check.hex(), hashed)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES PYDANTIC
# ═══════════════════════════════════════════════════════════════════════════════


class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=4, max_length=128)
    role: str = "viewer"
    display_name: str = ""


class UserLogin(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    username: str
    role: str
    display_name: str
    created_at: str
    is_active: bool


class UserUpdate(BaseModel):
    role: str | None = None
    display_name: str | None = None
    is_active: bool | None = None
    new_password: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

USERS_KEY_PREFIX = "hive:auth:user:"
VALID_ROLES = ("admin", "operator", "viewer")


class AuthService:
    """Service d'authentification avec stockage Redis."""

    def __init__(self, redis_client, jwt_secret: str, jwt_expiry_hours: int = 24):
        self.redis = redis_client
        self.jwt_secret = jwt_secret
        self.jwt_expiry_hours = jwt_expiry_hours

    async def init_default_admin(self):
        """Crée l'admin par défaut si aucun user n'existe."""
        admin_key = f"{USERS_KEY_PREFIX}admin"
        try:
            existing = await self.redis.get(admin_key)
            if existing is None:
                hashed, salt = hash_password("admin123")
                user_data = {
                    "username": "admin",
                    "password_hash": hashed,
                    "salt": salt,
                    "role": "admin",
                    "display_name": "Maître",
                    "created_at": datetime.now().isoformat(),
                    "is_active": True,
                }
                await self.redis.set(admin_key, json.dumps(user_data))
                logger.info("✅ Default admin user created (admin / admin123)")
            else:
                logger.info("✅ Admin user already exists")
        except Exception as e:
            logger.warning(f"⚠️ Could not init admin user (Redis unavailable?): {e}")

    async def login(self, username: str, password: str) -> TokenResponse | None:
        user_data = await self._get_user_data(username)
        if not user_data:
            return None
        if not user_data.get("is_active", True):
            return None
        if not verify_password(password, user_data["password_hash"], user_data["salt"]):
            return None

        payload = {
            "sub": username,
            "role": user_data["role"],
            "display_name": user_data.get("display_name", username),
            "iat": int(time.time()),
            "exp": int(time.time()) + self.jwt_expiry_hours * 3600,
        }
        token = create_jwt_token(payload, self.jwt_secret)

        return TokenResponse(
            access_token=token,
            user=UserInfo(
                username=username,
                role=user_data["role"],
                display_name=user_data.get("display_name", username),
                created_at=user_data.get("created_at", ""),
                is_active=user_data.get("is_active", True),
            ),
        )

    async def create_user(self, user: UserCreate) -> UserInfo | None:
        key = f"{USERS_KEY_PREFIX}{user.username}"
        existing = await self.redis.get(key)
        if existing:
            return None
        if user.role not in VALID_ROLES:
            return None

        hashed, salt = hash_password(user.password)
        user_data = {
            "username": user.username,
            "password_hash": hashed,
            "salt": salt,
            "role": user.role,
            "display_name": user.display_name or user.username,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
        }
        await self.redis.set(key, json.dumps(user_data))

        return UserInfo(
            username=user.username,
            role=user.role,
            display_name=user_data["display_name"],
            created_at=user_data["created_at"],
            is_active=True,
        )

    async def get_user(self, username: str) -> UserInfo | None:
        user_data = await self._get_user_data(username)
        if not user_data:
            return None
        return UserInfo(
            username=user_data["username"],
            role=user_data["role"],
            display_name=user_data.get("display_name", username),
            created_at=user_data.get("created_at", ""),
            is_active=user_data.get("is_active", True),
        )

    async def list_users(self) -> list[UserInfo]:
        users: list[UserInfo] = []
        try:
            keys = await self.redis._client.keys(f"{USERS_KEY_PREFIX}*")
            for key in keys:
                raw = await self.redis._client.get(key)
                if raw:
                    d = json.loads(raw)
                    users.append(
                        UserInfo(
                            username=d["username"],
                            role=d["role"],
                            display_name=d.get("display_name", ""),
                            created_at=d.get("created_at", ""),
                            is_active=d.get("is_active", True),
                        )
                    )
        except Exception as e:
            logger.error(f"Error listing users: {e}")
        return users

    async def update_user(self, username: str, update: UserUpdate) -> UserInfo | None:
        user_data = await self._get_user_data(username)
        if not user_data:
            return None

        if update.role is not None and update.role in VALID_ROLES:
            user_data["role"] = update.role
        if update.display_name is not None:
            user_data["display_name"] = update.display_name
        if update.is_active is not None:
            user_data["is_active"] = update.is_active
        if update.new_password is not None:
            hashed, salt = hash_password(update.new_password)
            user_data["password_hash"] = hashed
            user_data["salt"] = salt

        key = f"{USERS_KEY_PREFIX}{username}"
        await self.redis.set(key, json.dumps(user_data))

        return UserInfo(
            username=user_data["username"],
            role=user_data["role"],
            display_name=user_data.get("display_name", username),
            created_at=user_data.get("created_at", ""),
            is_active=user_data.get("is_active", True),
        )

    async def delete_user(self, username: str) -> bool:
        if username == "admin":
            return False
        key = f"{USERS_KEY_PREFIX}{username}"
        result = await self.redis._client.delete(key)
        return result > 0

    def validate_token(self, token: str) -> dict | None:
        return decode_jwt_token(token, self.jwt_secret)

    async def _get_user_data(self, username: str) -> dict | None:
        key = f"{USERS_KEY_PREFIX}{username}"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

security = HTTPBearer(auto_error=False)

_auth_service: AuthService | None = None


def set_auth_service(service: AuthService):
    global _auth_service
    _auth_service = service


def get_auth_service() -> AuthService:
    if _auth_service is None:
        raise RuntimeError("AuthService not initialized")
    return _auth_service


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict | None:
    """Extrait et valide le JWT. Retourne le payload ou None."""
    if credentials is None:
        return None
    auth = get_auth_service()
    return auth.validate_token(credentials.credentials)


async def require_auth(user: dict | None = Depends(get_current_user)) -> dict:
    """Exige un JWT valide."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(user: dict = Depends(require_auth)) -> dict:
    """Exige le rôle admin."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
