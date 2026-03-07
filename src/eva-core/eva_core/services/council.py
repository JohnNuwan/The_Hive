"""Routage des modeles LLM par role pour le Conseil EVA."""

from dataclasses import dataclass
import logging
import os
from typing import Optional

from shared import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRoute:
    """Route d'inference resolue pour un role."""

    role: str
    backend: str
    host: str
    port: int
    model: str


class CouncilService:
    """Routeur MoE applicatif: selectionne modele et endpoint selon le role."""

    def __init__(self) -> None:
        """Initialise les mappings role -> modele avec fallback sur la config."""
        self.settings = get_settings()
        self.current_model: Optional[str] = None
        self.current_role: str = "general"
        self.current_backend: str = self.settings.llm_backend

        self.role_aliases = {
            "builder": "code",
            "shadow": "research",
            "lab": "research",
            "sentinel": "research",
        }

        self.default_role_models = {
            "general": self.settings.council_model_general,
            "research": self.settings.council_model_research,
            "banker": self.settings.council_model_banker,
            "code": self.settings.council_model_code,
        }

    def _normalize_role(self, role: str) -> str:
        """Normalise le role demande vers un role routable."""
        role_key = (role or "general").strip().lower()
        role_key = self.role_aliases.get(role_key, role_key)
        if role_key not in self.default_role_models:
            return "general"
        return role_key

    def _resolve_backend(self, role: str) -> str:
        """Resout le backend cible pour un role (vllm/ollama)."""
        default_backend = self.settings.llm_backend.lower()
        role_backend = os.getenv(f"COUNCIL_BACKEND_{role.upper()}", "").strip().lower()
        if role_backend in {"vllm", "ollama"}:
            return role_backend
        return default_backend if default_backend in {"vllm", "ollama"} else "vllm"

    def _resolve_model(self, role: str) -> str:
        """Resout le modele cible pour un role donne."""
        role_model = os.getenv(f"COUNCIL_MODEL_{role.upper()}", "").strip()
        if role_model:
            return role_model
        return self.default_role_models.get(role, self.default_role_models["general"])

    def _resolve_endpoint(self, backend: str, role: str) -> tuple[str, int]:
        """Resout host/port pour un backend + role."""
        if backend == "ollama":
            host_default = self.settings.ollama_host
            port_default = self.settings.ollama_port
            host = os.getenv(
                f"COUNCIL_OLLAMA_HOST_{role.upper()}",
                os.getenv("COUNCIL_OLLAMA_HOST", host_default),
            )
            port_raw = os.getenv(
                f"COUNCIL_OLLAMA_PORT_{role.upper()}",
                os.getenv("COUNCIL_OLLAMA_PORT", str(port_default)),
            )
        else:
            host_default = self.settings.vllm_host
            port_default = self.settings.vllm_port
            host = os.getenv(
                f"COUNCIL_VLLM_HOST_{role.upper()}",
                os.getenv("COUNCIL_VLLM_HOST", host_default),
            )
            port_raw = os.getenv(
                f"COUNCIL_VLLM_PORT_{role.upper()}",
                os.getenv("COUNCIL_VLLM_PORT", str(port_default)),
            )

        try:
            port = int(port_raw)
        except ValueError:
            port = port_default

        return host, port

    def resolve_route(self, role: str = "general") -> ModelRoute:
        """Construit la route complete (backend/endpoint/modele) pour un role."""
        normalized_role = self._normalize_role(role)
        backend = self._resolve_backend(normalized_role)
        model = self._resolve_model(normalized_role)
        host, port = self._resolve_endpoint(backend, normalized_role)

        return ModelRoute(
            role=normalized_role,
            backend=backend,
            host=host,
            port=port,
            model=model,
        )

    async def prepare_model(self, role: str = "general") -> str:
        """Active la route d'un role et retourne le modele cible."""
        route = self.resolve_route(role)

        if (
            self.current_model != route.model
            or self.current_role != route.role
            or self.current_backend != route.backend
        ):
            logger.info(
                "COUNCIL route active: role=%s backend=%s model=%s endpoint=%s:%s",
                route.role,
                route.backend,
                route.model,
                route.host,
                route.port,
            )

        self.current_model = route.model
        self.current_role = route.role
        self.current_backend = route.backend
        return route.model

    async def unload_current(self) -> None:
        """Reinitialise l'etat courant du Conseil."""
        if self.current_model:
            logger.info("COUNCIL reset: role=%s model=%s", self.current_role, self.current_model)
        self.current_model = None
        self.current_role = "general"
        self.current_backend = self.settings.llm_backend


_council_service: Optional[CouncilService] = None


def get_council_service() -> CouncilService:
    """Retourne l'instance singleton du Conseil."""
    global _council_service
    if _council_service is None:
        _council_service = CouncilService()
    return _council_service

