"""
WebSocket Service — Communication temps réel avec le Dashboard Nexus
═══════════════════════════════════════════════════════════════════

Gère les connexions WebSocket bidirectionnelles pour :
- Diffuser les mises à jour en temps réel (trading, alertes, métriques)
- Recevoir des commandes depuis le Dashboard
- Notifier les événements critiques (Kill-Switch, Nemesis, etc.)
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketService:
    """Gestionnaire de connexions WebSocket pour la diffusion temps réel."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepte et enregistre une nouvelle connexion WebSocket."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(
            f"🔌 WebSocket connecté ({len(self.active_connections)} actif(s))"
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Retire une connexion WebSocket."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(
            f"🔌 WebSocket déconnecté ({len(self.active_connections)} actif(s))"
        )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Diffuse un message JSON à tous les clients connectés."""
        if not self.active_connections:
            return

        payload = json.dumps(message, default=str)
        disconnected = []

        for conn in self.active_connections:
            try:
                await conn.send_text(payload)
            except Exception:
                disconnected.append(conn)

        # Nettoyer les connexions mortes
        for conn in disconnected:
            await self.disconnect(conn)

    async def broadcast_event(self, event_type: str, data: Any) -> None:
        """Diffuse un événement typé (format standardisé pour le frontend)."""
        await self.broadcast({
            "type": event_type,
            "payload": data,
        })

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Envoie un message à un client spécifique."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception:
            await self.disconnect(websocket)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)
