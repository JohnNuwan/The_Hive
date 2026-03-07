"""Couche memoire long terme avec degradation propre si Mem0 est indisponible."""

import logging
import os
from typing import Any

try:
    from mem0 import Memory

    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    Memory = None

logger = logging.getLogger(__name__)


class MemoryLayer:
    """Gere la memoire long terme de maniere resiliente."""

    def __init__(self, user_id: str = "admin") -> None:
        """Initialise Mem0 si disponible, sinon active un mode degrade sans crash."""
        self.user_id = user_id
        self.memory = None
        self.enabled = False

        if not MEM0_AVAILABLE or Memory is None:
            logger.warning("Mem0 non installe. Memoire long terme desactivee.")
            return

        ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        ollama_base_url = f"http://{ollama_host}:{ollama_port}"

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "host.docker.internal"),
                    "port": int(os.getenv("QDRANT_PORT", 6333)),
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text:latest",
                    "ollama_base_url": ollama_base_url,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "gemma3",
                    "temperature": 0,
                    "ollama_base_url": ollama_base_url,
                },
            },
        }

        try:
            self.memory = Memory.from_config(config)
            self.enabled = True
            logger.info("Mem0 actif pour la memoire long terme.")
        except Exception as exc:
            logger.warning("Mem0 indisponible au demarrage (mode degrade): %s", exc)
            self.memory = None
            self.enabled = False

    def store_event(self, text: str, metadata: dict[str, Any] | None = None) -> bool:
        """Stocke un evenement si Mem0 est disponible."""
        if not self.enabled or self.memory is None:
            return False

        try:
            self.memory.add(text, user_id=self.user_id, metadata=metadata)
            return True
        except Exception as exc:
            logger.warning("Echec store_event Mem0 (ignore): %s", exc)
            return False

    def recall(self, query: str) -> list[dict[str, Any]]:
        """Retourne les souvenirs pertinents, ou une liste vide en mode degrade."""
        if not self.enabled or self.memory is None:
            return []

        try:
            return self.memory.search(query, user_id=self.user_id)
        except Exception as exc:
            logger.warning("Echec recall Mem0 (ignore): %s", exc)
            return []

    def get_user_profile(self) -> list[dict[str, Any]]:
        """Retourne le profil utilisateur construit par Mem0."""
        if not self.enabled or self.memory is None:
            return []

        try:
            return self.memory.get_all(user_id=self.user_id)
        except Exception as exc:
            logger.warning("Echec get_user_profile Mem0 (ignore): %s", exc)
            return []
