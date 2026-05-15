"""Client HTTP du relay central pour les agents followers."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from eva_banker.follower.config import FollowerAgentConfig
from eva_banker.follower.models import FollowerCommand, FollowerExecutionResult

logger = logging.getLogger(__name__)


class RelayClient:
    """Client asynchrone du relay de copy trading distribue."""

    def __init__(self, config: FollowerAgentConfig) -> None:
        """Initialise le client relay.

        Args:
            config (FollowerAgentConfig): Configuration locale.
        """

        self.config = config
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        """Ferme le client HTTP sous-jacent."""

        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        """Construit les entetes d'authentification relay.

        Returns:
            dict[str, str]: Entetes HTTP.
        """

        headers = {"X-Hive-Client-Id": self.config.client_id}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"
        return headers

    def _url(self, path: str) -> str:
        """Construit une URL absolue du relay.

        Args:
            path (str): Chemin API relatif.

        Returns:
            str: URL absolue.
        """

        return f"{self.config.relay_base_url.rstrip('/')}/{path.lstrip('/')}"

    async def fetch_commands(self, after: str | None = None) -> list[FollowerCommand]:
        """Recupere les commandes en attente pour ce client.

        Args:
            after (str | None): Derniere commande traitee.

        Returns:
            list[FollowerCommand]: Commandes parsees et normalisees.
        """

        params = {"client_id": self.config.client_id}
        if after:
            params["after"] = after
        response = await self._client.get(
            self._url("/api/follower/commands"),
            params=params,
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        raw_commands = payload.get("commands", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_commands, list):
            logger.warning("Relay: reponse commandes ignoree car non liste.")
            return []
        return [FollowerCommand(**item) for item in raw_commands if isinstance(item, dict)]

    async def acknowledge(self, result: FollowerExecutionResult) -> None:
        """Confirme le traitement d'une commande au relay.

        Args:
            result (FollowerExecutionResult): Resultat local.
        """

        payload = _model_to_dict(result)
        response = await self._client.post(
            self._url("/api/follower/ack"),
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()

    async def heartbeat(self, payload: dict[str, Any]) -> None:
        """Envoie l'etat courant de l'agent au relay.

        Args:
            payload (dict[str, Any]): Snapshot local serialisable.
        """

        response = await self._client.post(
            self._url("/api/follower/heartbeat"),
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Convertit un modele Pydantic en dictionnaire JSON-ready.

    Args:
        model (Any): Modele Pydantic v1 ou v2.

    Returns:
        dict[str, Any]: Donnees serialisables.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()
