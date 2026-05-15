"""Relay central minimal pour distribuer les commandes aux agents followers."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from eva_banker.follower.models import FollowerCommand, FollowerExecutionResult


class RelayPublishRequest(BaseModel):
    """Requete de publication d'une commande master vers des clients.

    Args:
        client_ids (list[str]): Clients cibles.
        command (FollowerCommand | None): Commande unique.
        commands (list[FollowerCommand]): Commandes multiples.
    """

    client_ids: list[str]
    command: FollowerCommand | None = None
    commands: list[FollowerCommand] = Field(default_factory=list)


class RelayStorage:
    """Stockage file-backed simple pour le relay follower."""

    def __init__(self, path: str | Path) -> None:
        """Initialise le stockage relay.

        Args:
            path (str | Path): Fichier JSON persistant.
        """

        self.path = Path(path)
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {
            "queues": {},
            "acks": {},
            "heartbeats": {},
            "created_at": datetime.now().isoformat(),
        }
        self._load()

    def publish(self, client_ids: list[str], commands: list[FollowerCommand]) -> dict[str, int]:
        """Ajoute des commandes dans les files clients.

        Args:
            client_ids (list[str]): Clients destinataires.
            commands (list[FollowerCommand]): Commandes a ajouter.

        Returns:
            dict[str, int]: Nombre de commandes publiees par client.
        """

        with self._lock:
            published: dict[str, int] = {}
            for client_id in client_ids:
                queue = self._payload.setdefault("queues", {}).setdefault(client_id, [])
                known_ids = {str(item.get("command_id")) for item in queue if isinstance(item, dict)}
                count = 0
                for command in commands:
                    command_id = str(command.command_id)
                    if command_id in known_ids:
                        continue
                    queue.append(_model_to_dict(command))
                    known_ids.add(command_id)
                    count += 1
                published[client_id] = count
            self._save_locked()
            return published

    def pending_commands(self, client_id: str, after: str | None = None) -> list[dict[str, Any]]:
        """Retourne les commandes non acquittees d'un client.

        Args:
            client_id (str): Client demandeur.
            after (str | None): Borne optionnelle fournie par l'agent.

        Returns:
            list[dict[str, Any]]: Commandes en attente.
        """

        with self._lock:
            queue = list(self._payload.get("queues", {}).get(client_id, []))
            acked = set(self._payload.get("acks", {}).get(client_id, {}).keys())
        all_pending = []
        pending = []
        after_seen = after is None
        for item in queue:
            command_id = str(item.get("command_id") or "")
            if not command_id or command_id in acked:
                continue
            all_pending.append(item)
            if not after_seen:
                after_seen = command_id == after
                continue
            pending.append(item)
        return pending if after_seen else all_pending

    def acknowledge(self, client_id: str, result: FollowerExecutionResult) -> None:
        """Enregistre un acquittement client.

        Args:
            client_id (str): Client emetteur.
            result (FollowerExecutionResult): Resultat local.
        """

        with self._lock:
            acks = self._payload.setdefault("acks", {}).setdefault(client_id, {})
            acks[result.command_id] = _model_to_dict(result)
            self._save_locked()

    def heartbeat(self, client_id: str, payload: dict[str, Any]) -> None:
        """Enregistre le dernier heartbeat client.

        Args:
            client_id (str): Client emetteur.
            payload (dict[str, Any]): Snapshot runtime.
        """

        with self._lock:
            heartbeat_payload = dict(payload)
            heartbeat_payload["received_at"] = datetime.now().isoformat()
            self._payload.setdefault("heartbeats", {})[client_id] = heartbeat_payload
            self._save_locked()

    def status(self) -> dict[str, Any]:
        """Retourne un statut synthetique du relay.

        Returns:
            dict[str, Any]: Etat des files, acks et heartbeats.
        """

        with self._lock:
            queues = self._payload.get("queues", {})
            acks = self._payload.get("acks", {})
            return {
                "clients": sorted(set(queues.keys()) | set(acks.keys()) | set(self._payload.get("heartbeats", {}).keys())),
                "queue_sizes": {client_id: len(items) for client_id, items in queues.items()},
                "ack_sizes": {client_id: len(items) for client_id, items in acks.items()},
                "heartbeats": dict(self._payload.get("heartbeats", {})),
                "updated_at": datetime.now().isoformat(),
            }

    def _load(self) -> None:
        """Charge le fichier persistant si disponible."""

        if not self.path.exists():
            return
        try:
            self._payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._payload["load_error_at"] = datetime.now().isoformat()

    def _save_locked(self) -> None:
        """Sauvegarde l'etat. Le verrou appelant doit deja etre tenu."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def create_app(storage: RelayStorage | None = None) -> FastAPI:
    """Construit l'application FastAPI du relay follower.

    Args:
        storage (RelayStorage | None): Stockage injecte pour tests.

    Returns:
        FastAPI: Application relay.
    """

    relay_storage = storage or RelayStorage(
        os.getenv("HIVE_FOLLOWER_RELAY_STATE_PATH", "data/follower_relay/state.json")
    )
    app = FastAPI(title="THE HIVE Follower Relay", version="0.1.0")

    @app.post("/api/master/commands")
    async def publish_commands(
        request: RelayPublishRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Publie des commandes master vers les clients followers."""

        _require_master_auth(authorization)
        commands = list(request.commands)
        if request.command is not None:
            commands.append(request.command)
        if not request.client_ids:
            raise HTTPException(status_code=400, detail="Aucun client cible.")
        if not commands:
            raise HTTPException(status_code=400, detail="Aucune commande a publier.")
        return {"published": relay_storage.publish(request.client_ids, commands)}

    @app.get("/api/follower/commands")
    async def get_commands(
        client_id: str = Query(...),
        after: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Retourne les commandes non acquittees d'un client."""

        _require_client_auth(client_id, authorization)
        return {"commands": relay_storage.pending_commands(client_id, after=after)}

    @app.post("/api/follower/ack")
    async def acknowledge(
        result: FollowerExecutionResult,
        x_hive_client_id: str = Header(...),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Acquitte une commande traitee par un follower."""

        _require_client_auth(x_hive_client_id, authorization)
        relay_storage.acknowledge(x_hive_client_id, result)
        return {"success": True}

    @app.post("/api/follower/heartbeat")
    async def heartbeat(
        payload: dict[str, Any],
        x_hive_client_id: str = Header(...),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Enregistre le heartbeat d'un follower."""

        _require_client_auth(x_hive_client_id, authorization)
        relay_storage.heartbeat(x_hive_client_id, payload)
        return {"success": True}

    @app.get("/api/relay/status")
    async def relay_status() -> dict[str, Any]:
        """Retourne l'etat du relay."""

        return relay_storage.status()

    return app


def _require_master_auth(authorization: str | None) -> None:
    """Valide le token master si configure."""

    expected = os.getenv("HIVE_FOLLOWER_RELAY_MASTER_TOKEN", "").strip()
    if expected and _extract_bearer(authorization) != expected:
        raise HTTPException(status_code=401, detail="Token master invalide.")


def _require_client_auth(client_id: str, authorization: str | None) -> None:
    """Valide le token client si configure."""

    tokens = _load_client_tokens()
    if not tokens:
        return
    expected = tokens.get(client_id) or tokens.get("*")
    if expected and _extract_bearer(authorization) != expected:
        raise HTTPException(status_code=401, detail="Token client invalide.")


def _load_client_tokens() -> dict[str, str]:
    """Charge les tokens clients depuis l'environnement.

    Returns:
        dict[str, str]: Mapping client_id -> token.
    """

    raw_json = os.getenv("HIVE_FOLLOWER_RELAY_CLIENT_TOKENS_JSON", "").strip()
    if raw_json:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return {str(key): str(value) for key, value in payload.items()}
    single = os.getenv("HIVE_FOLLOWER_RELAY_CLIENT_TOKEN", "").strip()
    return {"*": single} if single else {}


def _extract_bearer(authorization: str | None) -> str:
    """Extrait un bearer token HTTP.

    Args:
        authorization (str | None): Entete Authorization.

    Returns:
        str: Token extrait ou chaine vide.
    """

    raw = str(authorization or "").strip()
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Serialise un modele Pydantic v1 ou v2."""

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


app = create_app()


def main() -> None:
    """Lance le relay follower en mode uvicorn."""

    host = os.getenv("HIVE_FOLLOWER_RELAY_HOST", "0.0.0.0")
    port = int(os.getenv("HIVE_FOLLOWER_RELAY_PORT", "8705"))
    uvicorn.run("eva_banker.follower.relay_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
