"""Configuration locale de l'agent follower."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path("data/follower_agent/config.json")
DEFAULT_FLEET_CONFIG_PATH = Path("data/follower_agent/fleet.config.json")


class FollowerAgentConfig(BaseModel):
    """Configuration persistante d'un agent follower client.

    Args:
        client_id (str): Identifiant public du client cote relay.
        account_label (str): Nom lisible affiche dans l'interface.
        relay_base_url (str): URL du relay central.
        api_token (str): Jeton client transmis au relay.
        poll_interval_seconds (float): Frequence de lecture des commandes.
        heartbeat_interval_seconds (float): Frequence d'envoi du heartbeat.
        dry_run (bool): Simule l'execution sans envoyer d'ordre MT5.
        mock_mt5 (bool): Active le mode mock pour tests et demo.
        mt5_login (int): Login MT5 local.
        mt5_password (str): Mot de passe MT5 local.
        mt5_server (str): Serveur MT5 local.
        mt5_terminal_path (str): Chemin du terminal64.exe local.
        mt5_terminal_portable (bool): Active le mode portable MT5.
        allocation_ratio (float): Multiplicateur de risque local applique apres sizing dynamique.
        balance_reference (float | None): Capital de reference optionnel.
        master_balance_reference (float | None): Capital de reference du compte maitre.
        use_equity_for_sizing (bool): Utilise l'equity MT5 locale si disponible.
        symbol_map (dict[str, str]): Mapping symbole maitre -> symbole local.
        supported_symbols (list[str]): Univers local autorise.
        state_path (str): Fichier de liens tickets et idempotence.
        log_path (str): Fichier de log local.
    """

    client_id: str = "client-demo"
    account_label: str = "Follower Demo"
    relay_base_url: str = "http://127.0.0.1:8705"
    api_token: str = ""
    poll_interval_seconds: float = 1.0
    heartbeat_interval_seconds: float = 5.0
    dry_run: bool = True
    mock_mt5: bool = True
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_terminal_path: str = ""
    mt5_terminal_portable: bool = False
    allocation_ratio: float = 1.0
    balance_reference: float | None = None
    master_balance_reference: float | None = 10000.0
    use_equity_for_sizing: bool = True
    symbol_map: dict[str, str] = Field(default_factory=dict)
    supported_symbols: list[str] = Field(default_factory=list)
    state_path: str = "data/follower_agent/state.json"
    log_path: str = "logs/follower_agent.log"

    def to_safe_dict(self) -> dict[str, Any]:
        """Retourne la configuration sans secret lisible.

        Returns:
            dict[str, Any]: Configuration masquee pour affichage.
        """

        payload = _model_to_dict(self)
        if payload.get("api_token"):
            payload["api_token"] = "***"
        if payload.get("mt5_password"):
            payload["mt5_password"] = "***"
        return payload


class FollowerAccountConfig(FollowerAgentConfig):
    """Configuration d'un compte follower dans une flotte locale.

    Args:
        enabled (bool): Active ou ignore ce compte au demarrage global.
    """

    enabled: bool = True


class FollowerFleetConfig(BaseModel):
    """Configuration multi-comptes d'une app follower client.

    Args:
        fleet_id (str): Identifiant lisible de la flotte client.
        accounts (list[FollowerAccountConfig]): Comptes MT5 geres par l'app.
    """

    fleet_id: str = "fleet-demo"
    accounts: list[FollowerAccountConfig] = Field(
        default_factory=lambda: [
            FollowerAccountConfig(
                client_id="client-demo-1",
                account_label="Compte Demo 1",
                state_path="data/follower_agent/client-demo-1.state.json",
                log_path="logs/follower_agent_client-demo-1.log",
            )
        ]
    )

    def to_safe_dict(self) -> dict[str, Any]:
        """Retourne la flotte sans secret lisible.

        Returns:
            dict[str, Any]: Configuration masquee pour affichage.
        """

        return {
            "fleet_id": self.fleet_id,
            "accounts": [account.to_safe_dict() for account in self.accounts],
        }


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Serialise un modele Pydantic v1 ou v2.

    Args:
        model (BaseModel): Modele a convertir.

    Returns:
        dict[str, Any]: Donnees serialisables.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def load_follower_config(path: str | Path = DEFAULT_CONFIG_PATH) -> FollowerAgentConfig:
    """Charge la configuration locale de l'agent follower.

    Args:
        path (str | Path): Fichier JSON de configuration.

    Returns:
        FollowerAgentConfig: Configuration chargee ou configuration par defaut.
    """

    config_path = Path(path)
    if not config_path.exists():
        return FollowerAgentConfig()
    raw_payload = json.loads(config_path.read_text(encoding="utf-8"))
    return FollowerAgentConfig(**raw_payload)


def load_follower_fleet_config(path: str | Path = DEFAULT_FLEET_CONFIG_PATH) -> FollowerFleetConfig:
    """Charge la configuration multi-comptes follower.

    Args:
        path (str | Path): Fichier JSON de flotte.

    Returns:
        FollowerFleetConfig: Configuration multi-comptes chargee.
    """

    config_path = Path(path)
    if not config_path.exists():
        return FollowerFleetConfig()
    raw_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "accounts" not in raw_payload:
        single = FollowerAgentConfig(**raw_payload)
        return FollowerFleetConfig(
            fleet_id=single.client_id,
            accounts=[FollowerAccountConfig(**_model_to_dict(single))],
        )
    return FollowerFleetConfig(**raw_payload)


def save_follower_config(
    config: FollowerAgentConfig,
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    """Persiste la configuration locale de l'agent follower.

    Args:
        config (FollowerAgentConfig): Configuration a sauvegarder.
        path (str | Path): Fichier JSON cible.
    """

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _model_to_dict(config)
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_follower_fleet_config(
    config: FollowerFleetConfig,
    path: str | Path = DEFAULT_FLEET_CONFIG_PATH,
) -> None:
    """Persiste la configuration multi-comptes follower.

    Args:
        config (FollowerFleetConfig): Flotte a sauvegarder.
        path (str | Path): Fichier JSON cible.
    """

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _model_to_dict(config)
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
