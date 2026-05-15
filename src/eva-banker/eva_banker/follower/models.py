"""Modeles de protocole pour l'agent follower distribue."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from shared.models import TradeAction


class FollowerCommandType(str, Enum):
    """Type d'ordre recu par un agent follower."""

    OPEN = "open"
    CLOSE = "close"
    MODIFY = "modify"
    SYNC = "sync"
    PING = "ping"


class FollowerCommand(BaseModel):
    """Commande normalisee transmise par le relay central.

    Args:
        command_id (str): Identifiant idempotent de la commande.
        command_type (FollowerCommandType): Type d'action a executer.
        master_ticket (int | None): Ticket source sur le compte maitre.
        symbol (str | None): Symbole source avant mapping broker local.
        action (TradeAction | None): Direction BUY/SELL pour une ouverture.
        volume (Decimal | None): Volume maitre ou volume cible selon le relay.
        entry_price (Decimal | None): Prix d'entree indicatif.
        stop_loss (Decimal | None): Stop loss demande.
        take_profit (Decimal | None): Take profit demande.
        master_profit (Decimal | None): PnL realise sur le maitre a la cloture.
        close_reason (str | None): Raison de cloture source.
        full_close (bool): Force une fermeture totale locale.
        comment (str): Commentaire audit.
        payload (dict[str, Any]): Extension libre du protocole.
    """

    command_id: str
    command_type: FollowerCommandType
    master_ticket: int | None = None
    symbol: str | None = None
    action: TradeAction | None = None
    volume: Decimal | None = None
    entry_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    master_profit: Decimal | None = None
    close_reason: str | None = None
    full_close: bool = False
    comment: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Configuration de serialisation Pydantic."""

        json_encoders = {Decimal: str}


class FollowerExecutionResult(BaseModel):
    """Resultat local d'une commande follower."""

    command_id: str
    success: bool
    message: str
    ticket: int | None = None
    master_ticket: int | None = None
    local_symbol: str | None = None
    mode: str = "none"
    details: dict[str, Any] = Field(default_factory=dict)
    executed_at: datetime = Field(default_factory=datetime.now)


class FollowerRuntimeStatus(BaseModel):
    """Etat courant expose par l'agent et l'interface."""

    client_id: str
    account_label: str
    relay_connected: bool = False
    mt5_connected: bool = False
    running: bool = False
    paused: bool = False
    dry_run: bool = False
    processed_commands: int = 0
    linked_positions: int = 0
    last_error: str | None = None
    last_event: str | None = None
    updated_at: datetime = Field(default_factory=datetime.now)
