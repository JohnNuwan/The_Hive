"""
Client Redis Pub/Sub - Communication Inter-Agents THE HIVE
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable
from uuid import UUID

import redis.asyncio as redis
from pydantic import BaseModel

from shared.config import get_settings
from shared.models import AgentMessage, AgentMessageType

logger = logging.getLogger(__name__)


class UUIDEncoder(json.JSONEncoder):
    """Encoder JSON pour UUID et datetime"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class RedisClient:
    """Client Redis pour communication inter-agents"""

    def __init__(self, url: str | None = None):
        settings = get_settings()
        self.url = url or settings.redis_url
        self._client: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._subscribers: dict[str, list[Callable]] = {}

    async def connect(self) -> None:
        """Connexion à Redis"""
        if self._client is None:
            self._client = redis.from_url(self.url, decode_responses=True)
            await self._client.ping()
            logger.info(f"Connecté à Redis: {self.url}")

    async def disconnect(self) -> None:
        """Déconnexion de Redis"""
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.close()
            logger.info("Déconnecté de Redis")

    async def publish(self, channel: str, message: AgentMessage | dict) -> int:
        """Publie un message sur un channel"""
        await self.connect()
        if isinstance(message, BaseModel):
            data = message.model_dump()
        else:
            data = message

        json_data = json.dumps(data, cls=UUIDEncoder)
        result = await self._client.publish(channel, json_data)
        logger.debug(f"Publié sur {channel}: {message}")
        return result

    async def send_to_agent(
        self,
        source: str,
        target: str,
        action: str,
        payload: dict[str, Any] | None = None,
        msg_type: AgentMessageType = AgentMessageType.REQUEST,
        correlation_id: UUID | None = None,
    ) -> AgentMessage:
        """Envoie un message à un agent spécifique"""
        message = AgentMessage(
            type=msg_type,
            source_agent=source,
            target_agent=target,
            action=action,
            payload=payload or {},
            correlation_id=correlation_id,
        )
        channel = message.to_redis_channel()
        await self.publish(channel, message)
        return message

    async def subscribe(
        self,
        channels: list[str],
        callback: Callable[[str, dict], Any],
    ) -> None:
        """S'abonne à des channels avec callback"""
        await self.connect()

        if self._pubsub is None:
            self._pubsub = self._client.pubsub()

        for channel in channels:
            if channel not in self._subscribers:
                self._subscribers[channel] = []
            self._subscribers[channel].append(callback)

        await self._pubsub.subscribe(*channels)
        logger.info(f"Abonné aux channels: {channels}")

    async def listen(self) -> None:
        """Écoute les messages en continu"""
        if self._pubsub is None:
            raise RuntimeError("Pas d'abonnement actif")

        async for message in self._pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"]
                try:
                    data = json.loads(message["data"])
                    for callback in self._subscribers.get(channel, []):
                        await callback(channel, data)
                except json.JSONDecodeError:
                    logger.error(f"Message invalide sur {channel}: {message['data']}")
                except Exception as e:
                    logger.exception(f"Erreur callback sur {channel}: {e}")

    async def get(self, key: str) -> str | None:
        """Récupère une valeur Redis"""
        await self.connect()
        return await self._client.get(key)

    async def set(
        self,
        key: str,
        value: str | dict,
        ex: int | None = None,
    ) -> bool:
        """Définit une valeur Redis"""
        await self.connect()
        if isinstance(value, dict):
            value = json.dumps(value, cls=UUIDEncoder)
        return await self._client.set(key, value, ex=ex)

    async def cache_get(self, key: str) -> dict | None:
        """Récupère une valeur JSON du cache"""
        data = await self.get(key)
        if data:
            return json.loads(data)
        return None

    async def cache_set(
        self,
        key: str,
        value: dict,
        ttl_seconds: int = 300,
    ) -> bool:
        """Met en cache une valeur JSON"""
        return await self.set(key, value, ex=ttl_seconds)


# Instance globale
_redis_client: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """Retourne l'instance Redis globale"""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def init_redis() -> RedisClient:
    """Initialise la connexion Redis"""
    client = get_redis_client()
    await client.connect()
    return client
