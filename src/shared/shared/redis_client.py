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
        """
        Établit la connexion au serveur Redis.

        Initialise le client s'il n'existe pas encore et vérifie la connectivité (Ping).

        Raises:
            ConnectionError: Si le serveur Redis est inaccessible.
        """
        if self._client is None:
            self._client = redis.from_url(self.url, decode_responses=True)
            await self._client.ping()
            logger.info(f"Connecté à Redis: {self.url}")

    async def disconnect(self) -> None:
        """
        Ferme proprement la connexion Redis et les abonnements PubSub.
        """
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.close()
            logger.info("Déconnecté de Redis")

    async def publish(self, channel: str, message: AgentMessage | dict) -> int:
        """
        Publie un message sur un canal Redis spécifique.

        Args:
            channel (str): Le nom du canal (topic).
            message (AgentMessage | dict): L'objet ou dictionnaire à sérialiser.

        Returns:
            int: Le nombre de clients ayant reçu le message.
        """
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
        """
        Envoie un message structuré à un agent spécifique via le bus Redis.

        Args:
            source (str): L'ID de l'agent émetteur (ex: 'core').
            target (str): L'ID de l'agent destinataire (ex: 'banker').
            action (str): L'action demandée ou notifiée (ex: 'EXECUTE_ORDER').
            payload (dict | None): Les données associées au message.
            msg_type (AgentMessageType): Le type de message (Request, Alert...).
            correlation_id (UUID | None): ID de corrélation pour le tracking.

        Returns:
            AgentMessage: L'objet message complet qui a été envoyé.
        """
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

    async def broadcast_to_swarm(
        self,
        source: str,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """
        Diffuse un message à tous les agents connectés (Swarm Mode).

        Utilise le target spécial 'all' pour le broadcast.

        Args:
            source (str): L'agent émetteur.
            action (str): La commande ou l'événement broadcasté.
            payload (dict | None): Données contextuelles.

        Returns:
            AgentMessage: Le message broadcasté.
        """
        return await self.send_to_agent(
            source=source,
            target="all",
            action=action,
            payload=payload,
            msg_type=AgentMessageType.SWARM_COMMAND
        )

    async def subscribe(
        self,
        channels: list[str],
        callback: Callable[[str, dict], Any],
    ) -> None:
        """
        S'abonne à une liste de canaux et enregistre un callback pour le traitement.

        Args:
            channels (list[str]): Liste des topics Redis à écouter.
            callback (Callable): Fonction async à appeler à chaque message reçu.
                       Signature: `async def cb(channel: str, data: dict): ...`
        """
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
        """
        Boucle d'écoute infinie pour traiter les messages entrants.

        Cette méthode est bloquante et doit être lancée dans une tâche asyncio (background).
        Elle dispatch les messages vers les callbacks enregistrés.

        Raises:
            RuntimeError: Si aucun abonnement n'a été configuré avant l'appel.
        """
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
        """
        Récupère une valeur brute (string) depuis Redis.

        Args:
            key (str): La clé à interroger.

        Returns:
            str | None: La valeur si elle existe, sinon None.
        """
        await self.connect()
        return await self._client.get(key)

    async def set(
        self,
        key: str,
        value: str | dict,
        ex: int | None = None,
    ) -> bool:
        """
        Définit une valeur dans Redis (String ou JSON).

        Args:
            key (str): La clé de stockage.
            value (str | dict): La valeur (automatiquement sérialisée si dict).
            ex (int | None): Durée de vie en secondes (TTL).

        Returns:
            bool: True si l'opération a réussi.
        """
        await self.connect()
        if isinstance(value, dict):
            value = json.dumps(value, cls=UUIDEncoder)
        return await self._client.set(key, value, ex=ex)

    async def cache_get(self, key: str) -> dict | None:
        """
        Récupère et désérialise un objet JSON depuis le cache.

        Args:
            key (str): La clé du cache.

        Returns:
            dict | None: L'objet désérialisé ou None.
        """
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
        """
        Stocke un dictionnaire en tant que JSON avec une expiration.

        Args:
            key (str): La clé de cache.
            value (dict): L'objet à stocker.
            ttl_seconds (int): Durée de vie en secondes (défaut 300s).

        Returns:
            bool: True si succès.
        """
        return await self.set(key, value, ex=ttl_seconds)


# Instance globale
_redis_client: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """
    Retourne l'instance Singleton du client Redis.

    Crée l'instance si elle n'existe pas encore.

    Returns:
        RedisClient: L'instance partagée.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def init_redis() -> RedisClient:
    """
    Initialise la connexion du singleton Redis au démarrage de l'app.

    Returns:
        RedisClient: Le client connecté et prêt.
    """
    client = get_redis_client()
    await client.connect()
    return client
