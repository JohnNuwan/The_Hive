"""
Service Mémoire - Client Qdrant pour RAG
Gère le stockage et la recherche vectorielle des conversations
"""

import logging
from functools import lru_cache
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared import ChatMessage, get_settings

from eva_core.memory_layer import MemoryLayer

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Service de mémoire vectorielle avec Qdrant et Mem0.
    
    Permet:
    - Stockage des messages de conversation (Qdrant)
    - Recherche sémantique (RAG)
    - Mémoire adaptative long terme (Mem0)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "conversations",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client: AsyncQdrantClient | None = None
        self._embedding_dim = 768  # nomic-embed-text
        self.adaptive_memory = MemoryLayer()
        logger.info(f"MemoryService initialisé: {host}:{port}/{collection_name} + Mem0 Adaptive")

    async def _get_client(self) -> AsyncQdrantClient:
        """Retourne ou crée le client Qdrant"""
        if self._client is None:
            self._client = AsyncQdrantClient(host=self.host, port=self.port)
            await self._ensure_collection()
        return self._client

    async def _ensure_collection(self) -> None:
        """Crée la collection si elle n'existe pas ou si les dimensions ne matchent pas"""
        try:
            collections = await self._client.get_collections()
            existing = [c.name for c in collections.collections]

            if self.collection_name in existing:
                # Vérifier les dimensions
                info = await self._client.get_collection(self.collection_name)
                current_dim = info.config.params.vectors.size
                if current_dim != self._embedding_dim:
                    logger.warning(f"Dimension mismatch ({current_dim} vs {self._embedding_dim}). Recréation de la collection...")
                    await self._client.delete_collection(self.collection_name)
                    existing.remove(self.collection_name)

            if self.collection_name not in existing:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' créée (dim={self._embedding_dim})")
        except Exception as e:
            logger.warning(f"Qdrant non disponible: {e}")

    async def _embed_text(self, text: str) -> list[float]:
        """
        Génère un embedding réel pour le texte via Ollama/nomic-embed-text.
        """
        from langchain_ollama import OllamaEmbeddings
        from shared import get_settings
        
        settings = get_settings()
        
        try:
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=f"http://{settings.ollama_host}:{settings.ollama_port}"
            )
            # Utilisation de aembed_query pour l’async
            return await embeddings.aembed_query(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}. Fallback sur hash (danger).")
            # Fallback dégradé pour éviter de bloquer tout le système
            import hashlib
            import struct
            hash_bytes = hashlib.sha384(text.encode()).digest()
            floats = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i : i + 4]
                if len(chunk) == 4:
                    floats.append((struct.unpack("!f", chunk)[0] % 2.0) - 1.0)
            while len(floats) < self._embedding_dim:
                floats.append(0.0)
            return floats[: self._embedding_dim]

    async def store_message(self, message: ChatMessage) -> str:
        """Stocke un message dans la mémoire vectorielle"""
        try:
            client = await self._get_client()

            point_id = str(message.id)
            vector = await self._embed_text(message.content)

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "session_id": str(message.session_id),
                    "role": message.role.value,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "metadata": message.metadata,
                },
            )

            await client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            # Enrichissement de la mémoire adaptative (Mem0)
            self.adaptive_memory.store_event(message.content)

            logger.debug(f"Message stocké: {point_id}")
            return point_id

        except Exception as e:
            logger.warning(f"Erreur stockage mémoire: {e}")
            return ""

    def get_user_profile(self) -> list:
        """Récupère les préférences apprises par Mem0"""
        return self.adaptive_memory.get_user_profile()

    async def search(
        self,
        query: str,
        session_id: UUID | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Recherche sémantique dans la mémoire.
        """
        try:
            client = await self._get_client()
            query_vector = await self._embed_text(query)

            # Construire le filtre
            query_filter = None
            if session_id:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=str(session_id)),
                        )
                    ]
                )

            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )

            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "role": result.payload.get("role", ""),
                    "session_id": result.payload.get("session_id", ""),
                    "timestamp": result.payload.get("timestamp", ""),
                }
                for result in results
            ]

        except Exception as e:
            logger.warning(f"Erreur recherche mémoire: {e}")
            return []

    async def get_session_history(
        self,
        session_id: UUID,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Récupère l'historique d'une session"""
        try:
            client = await self._get_client()
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            results, _ = await client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=str(session_id)),
                        )
                    ]
                ),
                limit=limit,
            )

            # Trier par timestamp
            history = [
                {
                    "id": r.id,
                    "content": r.payload.get("content", ""),
                    "role": r.payload.get("role", ""),
                    "timestamp": r.payload.get("timestamp", ""),
                }
                for r in results
            ]
            history.sort(key=lambda x: x["timestamp"])

            return history

        except Exception as e:
            logger.warning(f"Erreur récupération historique: {e}")
            return []


@lru_cache
def get_memory_service() -> MemoryService:
    """Retourne l'instance mémoire configurée"""
    settings = get_settings()
    return MemoryService(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_conversations,
    )
