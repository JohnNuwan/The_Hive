import asyncio
import logging
import json
from typing import Any, Callable, Dict, Optional
try:
    from gmqtt import Client as MQTTClient
except ImportError:
    MQTTClient = None

logger = logging.getLogger(__name__)

class EVAMQTTClient:
    """
    Système Nerveux Secondaire MQTT pour E.V.A.
    Fournit la QoS (fiabilité) et le LWT (Last Will and Testament).
    """
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.client = None
        self._connected = asyncio.Event()
        self.subscriptions: Dict[str, Callable] = {}

    async def connect(self, host: str = "localhost", port: int = 1883):
        """Se connecte au broker MQTT avec un testament."""
        if not MQTTClient:
            logger.error("gmqtt non installé. MQTT désactivé.")
            return

        self.client = MQTTClient(self.agent_name)
        
        # Configuration du Testament (LWT)
        # Si l'agent se déconnecte brutalement, le broker publie ce message
        self.client.set_last_will(
            f"eva/status/{self.agent_name}", 
            payload=json.dumps({"status": "offline", "reason": "connection_lost"}),
            qos=1,
            retain=True
        )

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        try:
            await self.client.connect(host, port)
            await self._connected.wait()
            # Publier le statut online
            await self.publish(f"eva/status/{self.agent_name}", {"status": "online"}, qos=1, retain=True)
            logger.info(f"MQTT: Agent {self.agent_name} connecté au lien neural.")
        except Exception as e:
            logger.error(f"MQTT Connection failed: {e}")

    def _on_connect(self, client, flags, rc, properties):
        self._connected.set()
        logger.info("MQTT Neural Link: Connected.")

    def _on_disconnect(self, client, packet, exc):
        self._connected.clear()
        logger.warning("MQTT Neural Link: Disconnected.")

    def _on_message(self, client, topic, payload, qos, properties):
        if topic in self.subscriptions:
            try:
                data = json.loads(payload.decode())
                asyncio.create_task(self.subscriptions[topic](topic, data))
            except Exception as e:
                logger.error(f"MQTT Callback error on {topic}: {e}")

    async def subscribe(self, topic: str, callback: Callable):
        """S'abonne à un sujet avec une fonction de rappel."""
        self.subscriptions[topic] = callback
        if self.client:
            self.client.subscribe(topic, qos=1)
            logger.info(f"MQTT: Subscribed to {topic}")

    async def publish(self, topic: str, payload: Any, qos: int = 1, retain: bool = False):
        """Publie un message avec un niveau de QoS spécifique."""
        if self.client and self._connected.is_set():
            msg_payload = json.dumps(payload)
            self.client.publish(topic, msg_payload, qos=qos, retain=retain)
        else:
            logger.warning(f"MQTT: Cannot publish to {topic}, client not ready.")

    async def disconnect(self):
        if self.client:
            # Nettoyage du statut
            await self.publish(f"eva/status/{self.agent_name}", {"status": "offline", "reason": "graceful_shutdown"}, qos=1, retain=True)
            await self.client.disconnect()
