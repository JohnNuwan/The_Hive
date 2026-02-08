"""
Client MQTT — Système Nerveux Secondaire de THE HIVE.

Fournit un canal de communication redondant basé sur MQTT (QoS)
en complément de Redis Pub/Sub. Utilisé principalement pour :
- Les ordres critiques (Critical Path) avec QoS 2 (exactly-once).
- Le Last Will and Testament (LWT) pour la détection de pannes.
- La communication avec le Kernel Rust et les futurs composants IoT.

Dépendance optionnelle : gmqtt (pip install gmqtt).
Si gmqtt n'est pas installé, le client bascule en mode noop (silencieux).
"""

import asyncio
import logging
import json
from typing import Any, Callable

try:
    from gmqtt import Client as MQTTClient
except ImportError:
    MQTTClient = None

logger = logging.getLogger(__name__)

# Timeout de connexion MQTT en secondes (évite les blocages au démarrage)
MQTT_CONNECT_TIMEOUT = 5


class EVAMQTTClient:
    """
    Client MQTT asynchrone pour les agents de THE HIVE.

    Gère la connexion au broker, la publication/souscription de messages
    et le Last Will and Testament (détection de déconnexion brutale).

    Attributes:
        agent_name: Nom de l'agent (utilisé comme client ID MQTT).
        client: Instance du client gmqtt (ou None si non installé).
    """

    def __init__(self, agent_name: str):
        """
        Initialise le client MQTT pour un agent donné.

        Args:
            agent_name: Identifiant unique de l'agent (ex: 'core', 'banker').
        """
        self.agent_name = agent_name
        self.client = None
        self._connected = asyncio.Event()
        self.subscriptions: dict[str, Callable] = {}

    async def connect(self, host: str = "localhost", port: int = 1883):
        """
        Se connecte au broker MQTT avec un testament (LWT).

        Si gmqtt n'est pas installé ou si le broker n'est pas joignable,
        la connexion échoue silencieusement (mode dégradé).

        Le timeout de connexion est de 5 secondes pour ne pas bloquer
        le démarrage des services si Mosquitto n'est pas lancé.

        Args:
            host: Adresse du broker MQTT (défaut: localhost).
            port: Port du broker MQTT (défaut: 1883).
        """
        if not MQTTClient:
            logger.warning("gmqtt non installé. MQTT désactivé (mode dégradé).")
            return

        self.client = MQTTClient(self.agent_name)

        # Configuration du Testament (LWT)
        # Si l'agent se déconnecte brutalement, le broker publie ce message
        self.client.set_last_will(
            f"eva/status/{self.agent_name}",
            payload=json.dumps({"status": "offline", "reason": "connection_lost"}),
            qos=1,
            retain=True,
        )

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        try:
            await self.client.connect(host, port)
            # Timeout de connexion pour éviter le blocage
            await asyncio.wait_for(self._connected.wait(), timeout=MQTT_CONNECT_TIMEOUT)
            # Publier le statut online
            await self.publish(
                f"eva/status/{self.agent_name}",
                {"status": "online"},
                qos=1,
                retain=True,
            )
            logger.info(f"MQTT: Agent '{self.agent_name}' connecté au lien neural.")
        except asyncio.TimeoutError:
            logger.warning(
                f"MQTT: Timeout connexion ({MQTT_CONNECT_TIMEOUT}s). "
                f"Broker probablement non lancé — mode dégradé."
            )
            self.client = None
        except Exception as e:
            logger.warning(f"MQTT: Connexion échouée — {e} (mode dégradé)")
            self.client = None

    def _on_connect(self, client, flags, rc, properties):
        """Callback déclenché lors de la connexion réussie au broker."""
        self._connected.set()
        logger.info("MQTT Neural Link: Connected.")

    def _on_disconnect(self, client, packet, exc):
        """Callback déclenché lors de la déconnexion du broker."""
        self._connected.clear()
        logger.warning("MQTT Neural Link: Disconnected.")

    def _on_message(self, client, topic, payload, qos, properties):
        """
        Callback déclenché à la réception d'un message.

        Dispatch le message vers le callback enregistré pour le topic.
        """
        if topic in self.subscriptions:
            try:
                data = json.loads(payload.decode())
                asyncio.create_task(self.subscriptions[topic](topic, data))
            except Exception as e:
                logger.error(f"MQTT Callback error on {topic}: {e}")

    async def subscribe(self, topic: str, callback: Callable):
        """
        S'abonne à un sujet MQTT avec une fonction de rappel.

        Args:
            topic: Le sujet MQTT à écouter (ex: 'eva/banker/requests/critical').
            callback: Fonction async(topic, data) appelée à chaque message.
        """
        self.subscriptions[topic] = callback
        if self.client:
            self.client.subscribe(topic, qos=1)
            logger.info(f"MQTT: Subscribed to {topic}")

    async def publish(self, topic: str, payload: Any, qos: int = 1, retain: bool = False):
        """
        Publie un message sur un sujet MQTT.

        Args:
            topic: Le sujet de publication.
            payload: Les données à publier (sérialisées en JSON).
            qos: Niveau de qualité de service (0, 1 ou 2).
            retain: Si True, le broker conserve le dernier message.
        """
        if self.client and self._connected.is_set():
            msg_payload = json.dumps(payload)
            self.client.publish(topic, msg_payload, qos=qos, retain=retain)
        else:
            logger.debug(f"MQTT: Cannot publish to {topic}, client not ready.")

    async def disconnect(self):
        """Déconnexion propre du broker MQTT avec notification de statut."""
        if self.client:
            await self.publish(
                f"eva/status/{self.agent_name}",
                {"status": "offline", "reason": "graceful_shutdown"},
                qos=1,
                retain=True,
            )
            await self.client.disconnect()
