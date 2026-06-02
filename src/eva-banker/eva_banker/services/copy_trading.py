"""
Services de copy trading multi-instances pour The Banker.

Ce module implemente un routeur maitre -> instances filles. Le banker maitre
execute l'ordre sur son compte principal, puis le recopie vers d'autres
instances banker, chacune rattachee a son propre terminal MT5.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from typing import Any
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from shared import OrderSource, Position, TradeAction, TradeOrder, get_settings
from shared.internal_auth import get_internal_headers
from eva_banker.services.orion_bridge.bridge_helper import (
    SourceStrategy,
    OpenPayload,
    ClosePayload,
    BridgeConnector,
)

logger = logging.getLogger(__name__)


class CopyTradingTarget(BaseModel):
    """
    Decrit une instance banker cible pour le copy trading.

    Attributes:
        id (UUID): Identifiant stable de la cible.
        name (str): Nom lisible de l'instance cible.
        banker_base_url (str): URL de base de l'instance banker distante.
        allocation_ratio (Decimal): Ratio multiplicateur applique au volume.
        enabled (bool): Active ou non la cible.
        use_equity_for_sizing (bool): Utilise l'equity si True, sinon balance.
        balance_reference (Decimal | None): Capital de reference optionnel.
        broker (str): Broker rattache a la cible.
        server (str | None): Serveur MT5 associe si connu.
        login (int | None): Login du compte si connu.
        phase (str): Phase du compte prop firm.
        terminal_label (str | None): Etiquette descriptive du terminal.
        symbol_map (dict[str, str]): Traductions explicites symbole source -> cible.
        supported_symbols (list[str]): Univers de symboles accepte par la cible.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    banker_base_url: str
    allocation_ratio: Decimal = Decimal("1.0")
    enabled: bool = True
    use_equity_for_sizing: bool = True
    balance_reference: Decimal | None = None
    broker: str = "MT5"
    server: str | None = None
    login: int | None = None
    phase: str = "funded"
    terminal_label: str | None = None
    symbol_map: dict[str, str] = Field(default_factory=dict)
    supported_symbols: list[str] = Field(default_factory=list)


class RemoteTicketLink(BaseModel):
    """
    Relie un ticket maitre a un ticket enfant sur une instance distante.

    Attributes:
        target_id (UUID): Identifiant de la cible distante.
        remote_ticket (int): Ticket MT5 cree sur l'instance fille.
    """

    target_id: UUID
    remote_ticket: int


class CopyTradingRouter:
    """
    Routeur d'execution locale puis distante pour le copy trading.

    La classe preserve l'interface principale du service MT5 grace a une
    delegation dynamique. Seules les operations sensibles a la replication
    sont surchargees ici.
    """

    def __init__(self, primary_service: Any) -> None:
        """
        Initialise le routeur autour du compte principal.

        Args:
            primary_service (Any): Service d'execution local principal.
        """
        self.primary_service = primary_service
        self.settings = get_settings()
        self.targets: dict[UUID, CopyTradingTarget] = {}
        self._ticket_links: dict[int, list[RemoteTicketLink]] = {}
        self._target_symbols_cache: dict[UUID, tuple[float, dict[str, str]]] = {}
        self._lock = asyncio.Lock()
        self._winning_close_runner_ratio = Decimal("0.70")
        self._http_client = httpx.AsyncClient(
            timeout=max(2.0, float(self.settings.banker_copy_request_timeout_seconds))
        )
        self._target_symbols_cache_ttl_seconds = 300.0

    @staticmethod
    def _parse_login_override(raw_value: str | None) -> set[int]:
        """
        Convertit une liste d'identifiants en ensemble de logins MT5.

        Args:
            raw_value (str | None): Liste separee par des virgules, espaces ou points-virgules.

        Returns:
            set[int]: Logins MT5 valides.
        """
        if not raw_value:
            return set()

        normalized = str(raw_value).replace(";", ",").replace(" ", ",")
        logins: set[int] = set()
        for item in normalized.split(","):
            value = item.strip()
            if not value:
                continue
            try:
                logins.add(int(value))
            except ValueError:
                logger.warning("Copy trading: login ignore dans la surcharge: %s", value)
        return logins

    def _resolve_target_overrides(self) -> tuple[set[int], set[int]]:
        """
        Charge les surcharges runtime de cibles actives ou mises de cote.

        Returns:
            tuple[set[int], set[int]]: Logins forces actifs puis logins forces inactifs.
        """
        enabled_raw = os.getenv("BANKER_COPY_ENABLED_LOGINS") or str(
            getattr(self.settings, "banker_copy_enabled_logins", "") or ""
        )
        disabled_raw = os.getenv("BANKER_COPY_DISABLED_LOGINS") or str(
            getattr(self.settings, "banker_copy_disabled_logins", "") or ""
        )
        return self._parse_login_override(enabled_raw), self._parse_login_override(disabled_raw)

    def __getattr__(self, item: str) -> Any:
        """
        Delegue toute methode non surchargee au service principal.

        Args:
            item (str): Nom de l'attribut demande.

        Returns:
            Any: Attribut expose par le service principal.
        """
        return getattr(self.primary_service, item)

    async def initialize(self) -> None:
        """
        Charge les cibles configurees depuis les settings.
        """
        self.targets.clear()
        raw_payload = str(self.settings.banker_copy_targets_json or "").strip()
        if not raw_payload:
            logger.info("Copy trading: aucune cible configuree.")
            return

        try:
            entries = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            logger.error("Copy trading: configuration JSON invalide: %s", exc)
            return

        if not isinstance(entries, list):
            logger.error("Copy trading: la configuration doit etre une liste JSON.")
            return

        forced_enabled_logins, forced_disabled_logins = self._resolve_target_overrides()
        loaded_count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning("Copy trading: entree ignoree car non structurée.")
                continue
            try:
                target = CopyTradingTarget(**entry)
            except Exception as exc:
                logger.warning("Copy trading: cible ignoree car invalide: %s", exc)
                continue
            if target.login in forced_enabled_logins:
                target.enabled = True
                logger.info("Copy trading: cible forcee active pour le login %s.", target.login)
            if target.login in forced_disabled_logins:
                target.enabled = False
                logger.warning("Copy trading: cible mise de cote pour le login %s.", target.login)
            self.targets[target.id] = target
            loaded_count += 1

        logger.info("Copy trading: %s cible(s) chargee(s).", loaded_count)

    async def close(self) -> None:
        """
        Ferme proprement les ressources HTTP du routeur.
        """
        await self._http_client.aclose()

    async def disconnect(self) -> None:
        """
        Ferme le client HTTP puis deconnecte le service principal.
        """
        await self.close()
        await self.primary_service.disconnect()

    def get_targets_status(self) -> list[dict[str, Any]]:
        """
        Retourne l'etat des cibles de copy trading.

        Returns:
            list[dict[str, Any]]: Resume serialisable des cibles connues.
        """
        statuses: list[dict[str, Any]] = []
        for target in self.targets.values():
            statuses.append(
                {
                    "id": str(target.id),
                    "name": target.name,
                    "banker_base_url": target.banker_base_url,
                    "allocation_ratio": float(target.allocation_ratio),
                    "enabled": target.enabled,
                    "broker": target.broker,
                    "server": target.server,
                    "login": target.login,
                    "phase": target.phase,
                    "terminal_label": target.terminal_label,
                    "symbol_map": dict(target.symbol_map),
                    "supported_symbols": list(target.supported_symbols),
                }
            )
        return statuses

    async def get_targets_runtime_status(self) -> list[dict[str, Any]]:
        """
        Retourne un etat runtime des cibles de copy trading.

        Cette vue complete la configuration statique par un diagnostic reseau
        simple: disponibilite de l'instance distante, lecture du compte et
        nombre de positions visibles. Elle sert surtout au tableau de bord et
        au debogage des followers qui ratent une copie sans planter le master.

        Returns:
            list[dict[str, Any]]: Statut enrichi de chaque cible.
        """
        statuses: list[dict[str, Any]] = []
        for target in self.targets.values():
            runtime_status: dict[str, Any] = {
                "id": str(target.id),
                "name": target.name,
                "banker_base_url": target.banker_base_url,
                "allocation_ratio": float(target.allocation_ratio),
                "enabled": target.enabled,
                "broker": target.broker,
                "server": target.server,
                "login": target.login,
                "phase": target.phase,
                "terminal_label": target.terminal_label,
                "symbol_map": dict(target.symbol_map),
                "supported_symbols": list(target.supported_symbols),
                "reachable": False,
                "health_status": None,
                "mt5_connected": None,
                "balance": None,
                "equity": None,
                "positions_count": None,
                "symbols_count": None,
                "last_error": None,
            }

            try:
                health_response = await self._http_client.get(
                    self._join_url(target.banker_base_url, "/health"),
                    headers=get_internal_headers("banker"),
                )
                health_response.raise_for_status()
                health_payload = health_response.json()
                runtime_status["reachable"] = True
                runtime_status["health_status"] = health_payload.get("status")
                runtime_status["mt5_connected"] = health_payload.get("mt5_connected")
            except Exception as exc:
                runtime_status["last_error"] = f"health: {exc}"
                statuses.append(runtime_status)
                continue

            try:
                account_response = await self._http_client.get(
                    self._join_url(target.banker_base_url, "/account"),
                    headers=get_internal_headers("banker"),
                )
                account_response.raise_for_status()
                account_payload = account_response.json()
                runtime_status["balance"] = account_payload.get("balance")
                runtime_status["equity"] = account_payload.get("equity")
            except Exception as exc:
                runtime_status["last_error"] = f"account: {exc}"

            try:
                positions_response = await self._http_client.get(
                    self._join_url(target.banker_base_url, "/positions"),
                    headers=get_internal_headers("banker"),
                )
                positions_response.raise_for_status()
                positions_payload = positions_response.json()
                if isinstance(positions_payload, list):
                    runtime_status["positions_count"] = len(positions_payload)
            except Exception as exc:
                runtime_status["last_error"] = (
                    f"{runtime_status['last_error']} | positions: {exc}"
                    if runtime_status["last_error"]
                    else f"positions: {exc}"
                )

            try:
                symbols_response = await self._http_client.get(
                    self._join_url(target.banker_base_url, "/symbols/discover"),
                    headers=get_internal_headers("banker"),
                )
                symbols_response.raise_for_status()
                symbols_payload = symbols_response.json()
                raw_symbols = (
                    symbols_payload.get("symbols", []) if isinstance(symbols_payload, dict) else []
                )
                if isinstance(raw_symbols, list):
                    runtime_status["symbols_count"] = len(raw_symbols)
            except Exception as exc:
                runtime_status["last_error"] = (
                    f"{runtime_status['last_error']} | symbols: {exc}"
                    if runtime_status["last_error"]
                    else f"symbols: {exc}"
                )

            statuses.append(runtime_status)
        return statuses

    async def execute_order(self, order: TradeOrder) -> dict[str, Any]:
        """
        Execute l'ordre localement puis le recopie vers les instances filles.

        Si `account_id` cible directement une instance distante, l'ordre est
        route vers cette seule cible sans execution locale.

        Args:
            order (TradeOrder): Ordre maitre ou cible a executer.

        Returns:
            dict[str, Any]: Resultat local enrichi des details de replication.
        """
        if order.account_id is not None and order.account_id in self.targets:
            target = self.targets[order.account_id]
            return await self._execute_remote_target_order(target, order)

        local_result = await self.primary_service.execute_order(order)
        if order.source == OrderSource.COPY:
            return local_result
        if not local_result.get("success"):
            return local_result

        # NEW BRIDGE ------------------------------------------------
        try:
            source = SourceStrategy(broker_name="FTMO", account_size=100000)
            payload = OpenPayload(
                source_ticket_id=local_result["ticket"],
                symbol=order.symbol,
                volume=order.volume,
                type=order.action,
            )
            BridgeConnector().send_order(source_strategy=source, payload=payload)
        except Exception as e:
            logger.error(f"Bridge error: {e}")
        # NEW BRIDGE CALL ----------------------------------------

        active_targets = [target for target in self.targets.values() if target.enabled]
        if not active_targets:
            return local_result

        copy_results = await self._fan_out_order(order, local_result, active_targets)
        local_result["copy_results"] = copy_results
        local_result["copy_summary"] = {
            "targets": len(copy_results),
            "success": sum(1 for item in copy_results if item.get("success")),
            "failed": sum(1 for item in copy_results if not item.get("success")),
        }
        return local_result

    async def close_position(
        self,
        ticket: int,
        volume: Decimal | None = None,
    ) -> dict[str, Any]:
        """
        Ferme la position maitre puis tente de fermer les positions filles.

        Args:
            ticket (int): Ticket maitre local.
            volume (Decimal | None): Volume local a cloturer. Si absent, ferme
                la position complete.

        Returns:
            dict[str, Any]: Resultat local enrichi des retours de replication.
        """
        local_result = await self._close_local_position(ticket, volume=volume)
        if not local_result.get("success"):
            return local_result
        # NEW BRIDGE ----------------------------------------------------
        try:
            source = SourceStrategy(broker_name="FTMO", account_size=10000)
            full_close = volume is None
            payload = ClosePayload(
                source_ticket_id=str(ticket), full_close=full_close, reason="EXTERNAL_CLOSE"
            )
            BridgeConnector().send_order(source_strategy=source, payload=payload)
        except Exception as e:
            logger.error(f"BridgeError: {e}")

        # NEW BRIDGE ----------------------------------------

        remote_results = await self._close_remote_links(
            ticket,
            close_as_runner=self._should_keep_remote_runner(local_result),
        )
        if remote_results:
            local_result["copy_results"] = remote_results
        return local_result

    async def _close_local_position(
        self,
        ticket: int,
        volume: Decimal | None = None,
    ) -> dict[str, Any]:
        """
        Ferme une position locale avec preservation optionnelle d'un runner.

        Quand le maitre cloture integralement une position gagnante, le routeur
        tente de convertir cette cloture en prise partielle puis passage du
        reliquat au break-even. Cela permet de conserver un matelas de gains
        sur le compte source avant de repercuter le meme schema aux followers.

        Args:
            ticket (int): Ticket local a traiter.
            volume (Decimal | None): Volume explicite a fermer. Si renseigne,
                la demande est appliquee telle quelle sans logique runner.

        Returns:
            dict[str, Any]: Resultat de cloture locale.
        """
        if volume is not None:
            return await self.primary_service.close_position(ticket, volume=volume)

        position = await self._fetch_local_position_snapshot(ticket)
        if position is None:
            return await self.primary_service.close_position(ticket, volume=volume)

        if float(position.profit or 0) <= 0.0:
            result = await self.primary_service.close_position(ticket, volume=volume)
            result["copy_close_mode"] = "full_close"
            return result

        return await self._close_local_runner_position(position)

    async def _fetch_local_position_snapshot(self, ticket: int) -> Position | None:
        """
        Retourne le snapshot courant d'une position locale par ticket.

        Args:
            ticket (int): Ticket MT5 du compte maitre.

        Returns:
            Position | None: Position correspondante si elle existe encore.
        """
        try:
            positions = await self.primary_service.get_open_positions()
        except Exception as exc:
            logger.warning(
                "Copy trading: lecture locale impossible pour le ticket %s: %s",
                ticket,
                exc,
            )
            return None

        if not positions:
            return None

        for position in positions:
            if int(getattr(position, "ticket", 0) or 0) == ticket:
                return position
        return None

    async def _close_local_runner_position(self, position: Position) -> dict[str, Any]:
        """
        Transforme une cloture gagnante locale en prise partielle puis SL au BE.

        Args:
            position (Position): Snapshot local courant.

        Returns:
            dict[str, Any]: Resultat final de la sequence locale.
        """
        current_volume = Decimal(str(position.volume or "0"))
        break_even_price = float(position.open_price or 0)
        if current_volume <= Decimal("0") or break_even_price <= 0.0:
            result = await self.primary_service.close_position(position.ticket)
            result["copy_close_mode"] = "full_close"
            return result

        close_volume = (current_volume * self._winning_close_runner_ratio).quantize(
            Decimal("0.01"),
            rounding=ROUND_FLOOR,
        )
        remaining_volume = current_volume - close_volume
        if close_volume <= Decimal("0") or remaining_volume < Decimal("0.01"):
            result = await self.primary_service.close_position(position.ticket)
            result["copy_close_mode"] = "full_close"
            return result

        partial_result = await self.primary_service.close_position(
            position.ticket,
            volume=close_volume,
        )
        if not partial_result.get("success"):
            return partial_result

        # NEW BRIDGE ------------------------------------------------
        try:
            source = SourceStrategy(broker_name="FTMO", account_size=10000)
            payload = ClosePayload(
                source_ticket_id=str(position.ticket),
                full_close=False,
                reason="PARTIAL_PROFIT_CLOSE",
            )
            BridgeConnector().send_order(source_strategy=source, payload=payload)
        except Exception as e:
            logger.error(f"BridgeError: {e}")
        # NEW BRIDGE ------------------------------------------------

        remaining_volume = Decimal(str(partial_result.get("volume_remaining") or "0"))
        if remaining_volume <= Decimal("0"):
            partial_result["copy_close_mode"] = "full_close"
            partial_result["runner_mode"] = "full_close_fallback"
            return partial_result

        modify_result = await self.primary_service.modify_position(
            position.ticket,
            sl=break_even_price,
            tp=0.0,
        )
        if not modify_result.get("success"):
            fallback_result = await self.primary_service.close_position(position.ticket)
            partial_result["copy_close_mode"] = "full_close"
            partial_result["runner_mode"] = "break_even_full_close_fallback"
            partial_result["break_even_price"] = break_even_price
            partial_result["modify_result"] = modify_result
            partial_result["fallback_close_result"] = fallback_result
            partial_result["success"] = bool(partial_result.get("success")) and bool(
                fallback_result.get("success")
            )
            partial_result["message"] = (
                "Passage au break-even local impossible, reliquat cloture integralement."
            )
            return partial_result

        partial_result["copy_close_mode"] = "runner"
        partial_result["runner_mode"] = "partial_close_break_even"
        partial_result["break_even_price"] = break_even_price
        partial_result["modify_result"] = modify_result
        return partial_result

    async def modify_position(
        self, ticket: int, sl: float = 0.0, tp: float = 0.0
    ) -> dict[str, Any]:
        """
        Modifie une position maitre puis propage la modification aux copies.

        Args:
            ticket (int): Ticket maitre local.
            sl (float): Nouveau stop loss.
            tp (float): Nouveau take profit.

        Returns:
            dict[str, Any]: Resultat local enrichi des retours distants.
        """
        local_result = await self.primary_service.modify_position(ticket, sl=sl, tp=tp)
        remote_results = await self._modify_remote_links(ticket, sl=sl, tp=tp)
        if remote_results:
            local_result["copy_results"] = remote_results
        return local_result

    async def synchronize_external_close(
        self,
        ticket: int,
        profit: float,
        master_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Repercute une cloture deja survenue sur le compte maitre.

        Cette methode sert quand MT5 cloture la position source hors du
        chemin ``close_position`` du routeur (par exemple sur TP, SL ou
        fermeture manuelle directe). Le maitre etant deja ferme, seule la
        replication des followers reste a executer.

        Args:
            ticket (int): Ticket maitre deja cloture localement.
            profit (float): Profit final du ticket maitre.
            master_snapshot (dict[str, Any] | None): Dernier etat connu de la
                position maitre, utilise pour reconstruire les liens apres un
                redemarrage.

        Returns:
            dict[str, Any]: Resultat synthetique de la synchronisation.
        """
        if master_snapshot:
            await self._rebuild_links_for_external_close(ticket, master_snapshot)

        # TODO: NEW BRIDGE modify order ------------------------------------------------
        # try:
        # source = SourceStrategy(broker_name="FTMO", account_size=10000)
        # payload = ClosePayload(
        # source_ticket_id=str(ticket), full_close=True, reason="EXTERNAL_CLOSE"
        # )
        # BridgeConnector().send_order(source_strategy=source, payload=payload)
        # except Exception as e:
        # logger.error(f"BridgeError: {e}")

        # NEW BRIDGE -------------------------------------------------

        remote_results = await self._close_remote_links(
            ticket,
            close_as_runner=profit > 0.0,
        )
        return {
            "success": all(result.get("success", False) for result in remote_results)
            if remote_results
            else True,
            "ticket": ticket,
            "profit": profit,
            "copy_results": remote_results,
            "copy_summary": {
                "targets": len(remote_results),
                "success": sum(1 for item in remote_results if item.get("success")),
                "failed": sum(1 for item in remote_results if not item.get("success")),
            },
        }

    async def _rebuild_links_for_external_close(
        self,
        master_ticket: int,
        master_snapshot: dict[str, Any],
    ) -> None:
        """
        Reconstruit les liens d'une position maitre fermee hors routeur.

        Args:
            master_ticket (int): Ticket maitre source.
            master_snapshot (dict[str, Any]): Dernier snapshot connu du ticket.
        """
        async with self._lock:
            if self._ticket_links.get(master_ticket):
                return

        master_position = self._position_from_snapshot(master_ticket, master_snapshot)
        if master_position is None:
            return

        active_targets = [target for target in self.targets.values() if target.enabled]
        if not active_targets:
            return

        master_balance = await self._resolve_master_balance()
        for target in active_targets:
            resolved_symbol = await self._resolve_target_symbol(target, master_position.symbol)
            if not resolved_symbol:
                continue

            target_balance = await self._resolve_target_balance(target)
            expected_volume = self._scale_volume(
                master_volume=Decimal(str(master_position.volume)),
                master_balance=master_balance,
                target_balance=target_balance,
                allocation_ratio=target.allocation_ratio,
                symbol=master_position.symbol,
            )
            if expected_volume <= Decimal("0"):
                expected_volume = Decimal(str(master_position.volume))

            remote_positions = await self._fetch_remote_positions(target)
            matched_remote = self._match_remote_position(
                remote_positions=remote_positions,
                used_remote_tickets=set(),
                resolved_symbol=resolved_symbol,
                master_position=master_position,
                expected_volume=expected_volume,
            )
            if matched_remote is None:
                continue

            try:
                remote_ticket = int(matched_remote["ticket"])
            except (TypeError, ValueError):
                continue

            await self._register_remote_link(
                master_ticket=master_ticket,
                target_id=target.id,
                remote_ticket=remote_ticket,
            )

    @staticmethod
    def _position_from_snapshot(
        master_ticket: int,
        master_snapshot: dict[str, Any],
    ) -> Position | None:
        """
        Convertit un snapshot interne en position minimale.

        Args:
            master_ticket (int): Ticket maitre.
            master_snapshot (dict[str, Any]): Donnees sauvegardees par le moteur.

        Returns:
            Position | None: Position synthetique exploitable pour le matching.
        """
        try:
            symbol = str(master_snapshot.get("symbol") or "").strip()
            action = TradeAction(str(master_snapshot.get("action") or "").upper())
            entry_price = Decimal(str(master_snapshot.get("entry_price") or "0"))
            volume = Decimal(str(master_snapshot.get("volume") or "0.01"))
            open_time = master_snapshot.get("open_time")
            if isinstance(open_time, str):
                open_time = datetime.fromisoformat(open_time)
            if not isinstance(open_time, datetime):
                open_time = datetime.now()
        except Exception:
            return None

        if not symbol or entry_price <= Decimal("0") or volume <= Decimal("0"):
            return None

        return Position(
            ticket=master_ticket,
            symbol=symbol,
            action=action,
            volume=volume,
            open_price=entry_price,
            current_price=entry_price,
            stop_loss=Decimal(str(master_snapshot.get("stop_loss_price") or "0")) or None,
            take_profit=Decimal(str(master_snapshot.get("take_profit_price") or "0")) or None,
            profit=Decimal("0"),
            comment=str(master_snapshot.get("comment") or ""),
            open_time=open_time,
        )

    async def repair_open_positions(
        self,
        create_missing: bool = True,
        close_orphans: bool = False,
    ) -> dict[str, Any]:
        """
        Repare les liens de copy trading et rattrape les positions manquantes.

        Cette routine sert apres l'ajout de nouveaux followers ou apres un
        redemarrage controle du maitre. Elle inspecte les positions ouvertes
        du maitre, reconstruit les liens `master -> remote ticket` quand les
        followers ont deja la position, ouvre les positions absentes si
        `create_missing` reste actif, puis peut fermer uniquement des
        positions de reparation evidemment orphelines si `close_orphans`
        reste actif.

        Args:
            create_missing (bool): Si True, ouvre les positions absentes sur
                les followers actifs.
            close_orphans (bool): Si True, ferme seulement les positions
                followers de reparation manifestement orphelines.

        Returns:
            dict[str, Any]: Resume detaille de la reparation.
        """
        master_positions = await self.primary_service.get_open_positions() or []
        active_targets = [target for target in self.targets.values() if target.enabled]
        master_balance = await self._resolve_master_balance()
        master_ticket_ids = {int(position.ticket) for position in master_positions}

        async with self._lock:
            for master_ticket in master_ticket_ids:
                self._ticket_links.pop(master_ticket, None)

        summary: dict[str, Any] = {
            "success": True,
            "master_positions": len(master_positions),
            "active_targets": len(active_targets),
            "create_missing": create_missing,
            "close_orphans": close_orphans,
            "targets": [],
        }
        if not master_positions or not active_targets:
            return summary

        for target in active_targets:
            target_result = await self._repair_target_positions(
                target=target,
                master_positions=master_positions,
                master_balance=master_balance,
                create_missing=create_missing,
                close_orphans=close_orphans,
            )
            summary["targets"].append(target_result)
            if target_result.get("errors"):
                summary["success"] = False

        return summary

    async def _fan_out_order(
        self,
        master_order: TradeOrder,
        local_result: dict[str, Any],
        targets: list[CopyTradingTarget],
    ) -> list[dict[str, Any]]:
        """
        Recopie un ordre maitre vers toutes les cibles actives.

        Args:
            master_order (TradeOrder): Ordre execute sur le compte principal.
            local_result (dict[str, Any]): Resultat de l'ordre local.
            targets (list[CopyTradingTarget]): Cibles actives de replication.

        Returns:
            list[dict[str, Any]]: Resultats individuels par cible.
        """
        master_balance = await self._resolve_master_balance()
        tasks = [
            self._execute_scaled_copy(
                target=target, master_order=master_order, master_balance=master_balance
            )
            for target in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        normalized_results: list[dict[str, Any]] = []
        links: list[RemoteTicketLink] = []
        master_ticket = local_result.get("ticket")

        for target, result in zip(targets, results, strict=False):
            if isinstance(result, Exception):
                normalized_results.append(
                    {
                        "target_id": str(target.id),
                        "target_name": target.name,
                        "success": False,
                        "message": f"Echec copy trading: {result}",
                    }
                )
                continue

            normalized_results.append(result)
            if result.get("success") and result.get("ticket") is not None:
                links.append(
                    RemoteTicketLink(
                        target_id=target.id,
                        remote_ticket=int(result["ticket"]),
                    )
                )

        if master_ticket is not None and links:
            async with self._lock:
                self._ticket_links[int(master_ticket)] = links

        return normalized_results

    async def _repair_target_positions(
        self,
        target: CopyTradingTarget,
        master_positions: list[Position],
        master_balance: Decimal | None,
        create_missing: bool,
        close_orphans: bool,
    ) -> dict[str, Any]:
        """
        Repare les positions d'une cible follower pour le panier courant.

        Args:
            target (CopyTradingTarget): Cible follower a synchroniser.
            master_positions (list[Position]): Positions ouvertes du maitre.
            master_balance (Decimal | None): Capital courant du maitre.
            create_missing (bool): Ouvre les positions manquantes si True.
            close_orphans (bool): Ferme les positions followers orphelines si True.

        Returns:
            dict[str, Any]: Resultat detaille de la synchronisation.
        """
        remote_positions = await self._fetch_remote_positions(target)
        target_balance = await self._resolve_target_balance(target)
        used_remote_tickets: set[int] = set()
        repaired_links = 0
        created_positions = 0
        matched_positions = 0
        skipped_positions = 0
        missing_positions = 0
        orphan_positions = 0
        orphan_closed = 0
        unsupported_symbols: list[str] = []
        errors: list[str] = []
        details: list[dict[str, Any]] = []

        for master_position in master_positions:
            resolved_symbol = await self._resolve_target_symbol(target, master_position.symbol)
            if not resolved_symbol:
                skipped_positions += 1
                unsupported_symbols.append(master_position.symbol)
                details.append(
                    {
                        "master_ticket": int(master_position.ticket),
                        "symbol": master_position.symbol,
                        "status": "unsupported_symbol",
                    }
                )
                continue

            expected_volume = self._scale_volume(
                master_volume=Decimal(str(master_position.volume)),
                master_balance=master_balance,
                target_balance=target_balance,
                allocation_ratio=target.allocation_ratio,
                symbol=master_position.symbol,
            )
            if expected_volume <= Decimal("0"):
                skipped_positions += 1
                details.append(
                    {
                        "master_ticket": int(master_position.ticket),
                        "symbol": master_position.symbol,
                        "resolved_symbol": resolved_symbol,
                        "status": "zero_volume",
                    }
                )
                continue

            matched_remote = self._match_remote_position(
                remote_positions=remote_positions,
                used_remote_tickets=used_remote_tickets,
                resolved_symbol=resolved_symbol,
                master_position=master_position,
                expected_volume=expected_volume,
            )
            if matched_remote is not None:
                remote_ticket = int(matched_remote["ticket"])
                used_remote_tickets.add(remote_ticket)
                await self._register_remote_link(
                    master_ticket=int(master_position.ticket),
                    target_id=target.id,
                    remote_ticket=remote_ticket,
                )
                repaired_links += 1
                matched_positions += 1
                details.append(
                    {
                        "master_ticket": int(master_position.ticket),
                        "remote_ticket": remote_ticket,
                        "symbol": master_position.symbol,
                        "resolved_symbol": resolved_symbol,
                        "status": "matched_existing",
                    }
                )
                continue

            missing_positions += 1
            if not create_missing:
                details.append(
                    {
                        "master_ticket": int(master_position.ticket),
                        "symbol": master_position.symbol,
                        "resolved_symbol": resolved_symbol,
                        "status": "missing_remote",
                    }
                )
                continue

            repair_order = self._build_repair_order(
                master_position=master_position,
                resolved_symbol=resolved_symbol,
                volume=expected_volume,
            )
            created_result = await self._execute_remote_target_order(target, repair_order)
            if created_result.get("success") and created_result.get("ticket") is not None:
                remote_ticket = int(created_result["ticket"])
                used_remote_tickets.add(remote_ticket)
                await self._register_remote_link(
                    master_ticket=int(master_position.ticket),
                    target_id=target.id,
                    remote_ticket=remote_ticket,
                )
                repaired_links += 1
                created_positions += 1
                details.append(
                    {
                        "master_ticket": int(master_position.ticket),
                        "remote_ticket": remote_ticket,
                        "symbol": master_position.symbol,
                        "resolved_symbol": resolved_symbol,
                        "status": "created_missing",
                    }
                )
                continue

            error_message = str(created_result.get("message") or "Creation distante inconnue.")
            errors.append(f"{master_position.symbol}: {error_message}")
            details.append(
                {
                    "master_ticket": int(master_position.ticket),
                    "symbol": master_position.symbol,
                    "resolved_symbol": resolved_symbol,
                    "status": "create_failed",
                    "message": error_message,
                }
            )

        if close_orphans:
            orphan_result = await self._close_orphan_remote_positions(
                target=target,
                remote_positions=remote_positions,
                used_remote_tickets=used_remote_tickets,
            )
            orphan_positions = int(orphan_result.get("orphan_positions", 0) or 0)
            orphan_closed = int(orphan_result.get("orphan_closed", 0) or 0)
            errors.extend(list(orphan_result.get("errors") or []))
            details.extend(list(orphan_result.get("details") or []))

        return {
            "target_id": str(target.id),
            "target_name": target.name,
            "login": target.login,
            "matched_positions": matched_positions,
            "created_positions": created_positions,
            "repaired_links": repaired_links,
            "missing_positions": missing_positions,
            "orphan_positions": orphan_positions,
            "orphan_closed": orphan_closed,
            "skipped_positions": skipped_positions,
            "unsupported_symbols": unsupported_symbols,
            "errors": errors,
            "details": details,
        }

    async def _close_orphan_remote_positions(
        self,
        *,
        target: CopyTradingTarget,
        remote_positions: list[dict[str, Any]],
        used_remote_tickets: set[int],
    ) -> dict[str, Any]:
        """
        Ferme les positions followers orphelines creees par une ancienne
        reparation automatique.

        Args:
            target (CopyTradingTarget): Cible follower a nettoyer.
            remote_positions (list[dict[str, Any]]): Positions ouvertes distantes.
            used_remote_tickets (set[int]): Tickets deja relies a un ticket maitre.

        Returns:
            dict[str, Any]: Resume des fermetures orphelines.
        """
        orphan_positions = 0
        orphan_closed = 0
        errors: list[str] = []
        details: list[dict[str, Any]] = []

        for remote_position in remote_positions:
            try:
                remote_ticket = int(remote_position.get("ticket") or 0)
            except (TypeError, ValueError):
                continue
            if remote_ticket <= 0 or remote_ticket in used_remote_tickets:
                continue
            if not self._is_copy_managed_remote_position(remote_position):
                continue

            orphan_positions += 1
            close_result = await self._close_remote_ticket(target, remote_ticket)
            if close_result.get("success"):
                orphan_closed += 1
                details.append(
                    {
                        "remote_ticket": remote_ticket,
                        "symbol": str(remote_position.get("symbol") or ""),
                        "status": "closed_orphan",
                    }
                )
                continue

            error_message = str(close_result.get("message") or "Cloture orpheline inconnue.")
            errors.append(f"{remote_ticket}: {error_message}")
            details.append(
                {
                    "remote_ticket": remote_ticket,
                    "symbol": str(remote_position.get("symbol") or ""),
                    "status": "close_orphan_failed",
                    "message": error_message,
                }
            )

        return {
            "orphan_positions": orphan_positions,
            "orphan_closed": orphan_closed,
            "errors": errors,
            "details": details,
        }

    @staticmethod
    def _is_copy_managed_remote_position(remote_position: dict[str, Any]) -> bool:
        """
        Indique si une position distante ressemble a un artefact de reparation.

        Args:
            remote_position (dict[str, Any]): Position distante serialisee.

        Returns:
            bool: True si le commentaire indique une ancienne reparation
                automatique devenue orpheline.
        """
        comment = str(remote_position.get("comment") or "").strip().upper()
        if not comment:
            return False
        return comment.startswith("COPY") or comment.startswith("EVA CLOSE")

    async def _execute_scaled_copy(
        self,
        target: CopyTradingTarget,
        master_order: TradeOrder,
        master_balance: Decimal | None,
    ) -> dict[str, Any]:
        """
        Calcule le volume proportionnel puis envoie l'ordre a une cible.

        Args:
            target (CopyTradingTarget): Cible distante.
            master_order (TradeOrder): Ordre maitre.
            master_balance (Decimal | None): Capital de reference du maitre.

        Returns:
            dict[str, Any]: Resultat de l'execution distante.
        """
        target_balance = await self._resolve_target_balance(target)
        scaled_volume = self._scale_volume(
            master_volume=master_order.volume,
            master_balance=master_balance,
            target_balance=target_balance,
            allocation_ratio=target.allocation_ratio,
            symbol=master_order.symbol,
        )
        if scaled_volume <= Decimal("0"):
            return {
                "target_id": str(target.id),
                "target_name": target.name,
                "success": False,
                "message": "Volume copie nul apres proportionalite.",
            }

        resolved_symbol = await self._resolve_target_symbol(target, master_order.symbol)
        if not resolved_symbol:
            return {
                "target_id": str(target.id),
                "target_name": target.name,
                "success": False,
                "message": f"Symbole {master_order.symbol} incompatible avec la cible {target.name}.",
            }

        copy_order = master_order.model_copy(
            update={
                "symbol": resolved_symbol,
                "volume": scaled_volume,
                "source": OrderSource.COPY,
                "account_id": None,
                "comment": self._build_copy_comment(master_order.comment),
            }
        )
        result = await self._execute_remote_target_order(target, copy_order)
        result.setdefault("target_id", str(target.id))
        result.setdefault("target_name", target.name)
        result["scaled_volume"] = float(scaled_volume)
        result["resolved_symbol"] = resolved_symbol
        return result

    async def _execute_remote_target_order(
        self,
        target: CopyTradingTarget,
        order: TradeOrder,
    ) -> dict[str, Any]:
        """
        Execute un ordre sur une instance banker distante.

        Args:
            target (CopyTradingTarget): Cible distante.
            order (TradeOrder): Ordre a transmettre.

        Returns:
            dict[str, Any]: Resultat HTTP normalise.
        """
        payload = {
            "symbol": order.symbol,
            "action": order.action.value
            if isinstance(order.action, TradeAction)
            else str(order.action),
            "volume": str(order.volume),
            "entry_price": str(order.entry_price) if order.entry_price is not None else None,
            "stop_loss": str(order.stop_loss_price) if order.stop_loss_price is not None else None,
            "take_profit": str(order.take_profit_price)
            if order.take_profit_price is not None
            else None,
            "account_id": str(order.account_id) if order.account_id is not None else None,
            "comment": order.comment,
            "source": order.source.value if hasattr(order.source, "value") else str(order.source),
        }
        try:
            response = await self._http_client.post(
                self._join_url(target.banker_base_url, "/orders"),
                headers=get_internal_headers("banker"),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(
                "Copy trading: execution distante impossible sur %s pour %s %s %s: %s",
                target.name,
                order.symbol,
                order.action.value if isinstance(order.action, TradeAction) else str(order.action),
                order.volume,
                exc,
            )
            return {
                "target_id": str(target.id),
                "target_name": target.name,
                "success": False,
                "message": f"Execution distante impossible: {exc}",
            }

        data["target_id"] = str(target.id)
        data["target_name"] = target.name
        log_payload = "Copy trading: cible=%s | ordre=%s %s %s | succes=%s | ticket=%s | message=%s"
        log_args = (
            target.name,
            order.symbol,
            order.action.value if isinstance(order.action, TradeAction) else str(order.action),
            order.volume,
            data.get("success"),
            data.get("ticket"),
            data.get("message"),
        )
        if data.get("success"):
            logger.info(log_payload, *log_args)
        else:
            logger.warning(log_payload, *log_args)
        return data

    def _should_keep_remote_runner(self, local_result: dict[str, Any]) -> bool:
        """
        Indique si une cloture gagnante du maitre doit devenir un runner distant.

        Args:
            local_result (dict[str, Any]): Resultat de cloture du maitre.

        Returns:
            bool: True si les followers doivent conserver un reliquat au BE.
        """
        close_mode = str(local_result.get("copy_close_mode") or "").strip().lower()
        if close_mode == "runner":
            return True
        if close_mode == "full_close":
            return False
        try:
            return float(local_result.get("profit", 0) or 0) > 0.0
        except (TypeError, ValueError):
            return False

    async def _close_remote_links(
        self,
        master_ticket: int,
        close_as_runner: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ferme les tickets distants lies a un ticket maitre.

        Args:
            master_ticket (int): Ticket maitre local.
            close_as_runner (bool): Si True, ferme la moitie de la position
                distante puis deplace le stop au break-even.

        Returns:
            list[dict[str, Any]]: Resultats de cloture par cible.
        """
        async with self._lock:
            links = list(self._ticket_links.get(master_ticket, []))

        if not links:
            return []

        tasks = []
        for link in links:
            target = self.targets.get(link.target_id)
            if target is None or not target.enabled:
                continue
            if close_as_runner:
                tasks.append(self._close_remote_runner_ticket(target, link.remote_ticket))
            else:
                tasks.append(self._close_remote_ticket(target, link.remote_ticket))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        normalized: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                normalized.append(
                    {"success": False, "message": f"Echec cloture distante: {result}"}
                )
            else:
                normalized.append(result)

        async with self._lock:
            self._ticket_links.pop(master_ticket, None)

        return normalized

    async def _close_remote_runner_ticket(
        self,
        target: CopyTradingTarget,
        remote_ticket: int,
    ) -> dict[str, Any]:
        """
        Ferme partiellement un follower puis place le stop restant au break-even.

        Args:
            target (CopyTradingTarget): Cible distante.
            remote_ticket (int): Ticket distant a traiter.

        Returns:
            dict[str, Any]: Resultat composite de la cloture partielle et du
                passage au break-even.
        """
        position_snapshot = await self._fetch_remote_position_snapshot(target, remote_ticket)
        if position_snapshot is None:
            return await self._close_remote_ticket(target, remote_ticket)

        try:
            current_volume = Decimal(str(position_snapshot.get("volume") or "0"))
            break_even_price = float(position_snapshot.get("open_price") or 0.0)
        except (ArithmeticError, TypeError, ValueError):
            return await self._close_remote_ticket(target, remote_ticket)

        if current_volume <= Decimal("0") or break_even_price <= 0.0:
            return await self._close_remote_ticket(target, remote_ticket)

        close_volume = (current_volume * self._winning_close_runner_ratio).quantize(
            Decimal("0.01"),
            rounding=ROUND_FLOOR,
        )
        remaining_volume = current_volume - close_volume
        if close_volume <= Decimal("0") or remaining_volume < Decimal("0.01"):
            return await self._close_remote_ticket(target, remote_ticket)

        partial_result = await self._close_remote_ticket(
            target,
            remote_ticket,
            volume=close_volume,
        )
        if not partial_result.get("success"):
            return await self._close_remote_ticket(target, remote_ticket)

        remaining_volume = Decimal(str(partial_result.get("volume_remaining") or "0"))
        if remaining_volume <= Decimal("0"):
            partial_result["runner_mode"] = "full_close_fallback"
            return partial_result

        modify_result = await self._modify_remote_ticket(
            target,
            remote_ticket,
            sl=break_even_price,
            tp=0.0,
        )
        partial_result["runner_mode"] = "partial_close_break_even"
        partial_result["break_even_price"] = break_even_price
        partial_result["modify_result"] = modify_result
        partial_result["success"] = bool(partial_result.get("success")) and bool(
            modify_result.get("success")
        )
        if not modify_result.get("success"):
            partial_result["message"] = (
                "Cloture partielle distante reussie, mais passage au break-even echoue."
            )
        return partial_result

    async def _modify_remote_links(
        self, master_ticket: int, sl: float, tp: float
    ) -> list[dict[str, Any]]:
        """
        Propage une modification de SL/TP aux tickets distants lies.

        Args:
            master_ticket (int): Ticket maitre local.
            sl (float): Nouveau stop loss.
            tp (float): Nouveau take profit.

        Returns:
            list[dict[str, Any]]: Resultats de modification par cible.
        """
        async with self._lock:
            links = list(self._ticket_links.get(master_ticket, []))

        if not links:
            return []

        tasks = []
        for link in links:
            target = self.targets.get(link.target_id)
            if target is None or not target.enabled:
                continue
            tasks.append(self._modify_remote_ticket(target, link.remote_ticket, sl=sl, tp=tp))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        normalized: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                normalized.append(
                    {"success": False, "message": f"Echec modification distante: {result}"}
                )
            else:
                normalized.append(result)
        return normalized

    async def _fetch_remote_position_snapshot(
        self,
        target: CopyTradingTarget,
        remote_ticket: int,
    ) -> dict[str, Any] | None:
        """
        Lit une position distante pour recuperer son volume et son prix d'entree.

        Args:
            target (CopyTradingTarget): Cible distante.
            remote_ticket (int): Ticket distant a rechercher.

        Returns:
            dict[str, Any] | None: Position distante si elle existe, sinon None.
        """
        try:
            response = await self._http_client.get(
                self._join_url(target.banker_base_url, "/positions"),
                headers=get_internal_headers("banker"),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.info(
                "Copy trading: lecture de la position distante impossible sur %s pour %s: %s",
                target.name,
                remote_ticket,
                exc,
            )
            return None

        if not isinstance(payload, list):
            return None

        for position in payload:
            if not isinstance(position, dict):
                continue
            try:
                if int(position.get("ticket")) == int(remote_ticket):
                    return position
            except (TypeError, ValueError):
                continue
        return None

    async def _fetch_remote_positions(self, target: CopyTradingTarget) -> list[dict[str, Any]]:
        """
        Lit toutes les positions ouvertes sur une cible follower.

        Args:
            target (CopyTradingTarget): Cible distante a interroger.

        Returns:
            list[dict[str, Any]]: Positions ouvertes vues par l'instance cible.
        """
        try:
            response = await self._http_client.get(
                self._join_url(target.banker_base_url, "/positions"),
                headers=get_internal_headers("banker"),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.info(
                "Copy trading: lecture des positions distante impossible sur %s: %s",
                target.name,
                exc,
            )
            return []
        return payload if isinstance(payload, list) else []

    async def _close_remote_ticket(
        self,
        target: CopyTradingTarget,
        remote_ticket: int,
        volume: Decimal | None = None,
    ) -> dict[str, Any]:
        """
        Ferme une position distante sur une instance fille.

        Args:
            target (CopyTradingTarget): Cible distante.
            remote_ticket (int): Ticket a cloturer.
            volume (Decimal | None): Volume a cloturer. Si absent, ferme toute
                la position distante.

        Returns:
            dict[str, Any]: Resultat HTTP normalise.
        """
        try:
            response = await self._http_client.delete(
                self._join_url(target.banker_base_url, f"/positions/{remote_ticket}"),
                headers=get_internal_headers("banker"),
                params={"volume": float(volume)} if volume is not None else None,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {
                "target_id": str(target.id),
                "target_name": target.name,
                "success": False,
                "message": f"Cloture distante impossible: {exc}",
            }
        data["target_id"] = str(target.id)
        data["target_name"] = target.name
        return data

    async def _modify_remote_ticket(
        self,
        target: CopyTradingTarget,
        remote_ticket: int,
        sl: float,
        tp: float,
    ) -> dict[str, Any]:
        """
        Modifie un ticket distant sur une instance fille.

        Args:
            target (CopyTradingTarget): Cible distante.
            remote_ticket (int): Ticket a modifier.
            sl (float): Nouveau stop loss.
            tp (float): Nouveau take profit.

        Returns:
            dict[str, Any]: Resultat HTTP normalise.
        """
        try:
            response = await self._http_client.post(
                self._join_url(target.banker_base_url, f"/positions/{remote_ticket}/modify"),
                headers=get_internal_headers("banker"),
                json={"stop_loss": sl, "take_profit": tp},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {
                "target_id": str(target.id),
                "target_name": target.name,
                "success": False,
                "message": f"Modification distante impossible: {exc}",
            }
        data["target_id"] = str(target.id)
        data["target_name"] = target.name
        return data

    async def _register_remote_link(
        self,
        master_ticket: int,
        target_id: UUID,
        remote_ticket: int,
    ) -> None:
        """
        Enregistre un lien maitre -> follower en evitant les doublons.

        Args:
            master_ticket (int): Ticket maitre.
            target_id (UUID): Identifiant de la cible follower.
            remote_ticket (int): Ticket distant rattache.
        """
        async with self._lock:
            links = list(self._ticket_links.get(master_ticket, []))
            deduped_links = [
                link
                for link in links
                if not (link.target_id == target_id and link.remote_ticket == remote_ticket)
            ]
            deduped_links.append(RemoteTicketLink(target_id=target_id, remote_ticket=remote_ticket))
            self._ticket_links[master_ticket] = deduped_links

    async def _resolve_master_balance(self) -> Decimal | None:
        """
        Recupere le capital de reference du compte principal.

        Returns:
            Decimal | None: Equity ou balance du compte principal.
        """
        account = await self.primary_service.get_account_info()
        if account is None:
            return None
        return Decimal(str(account.equity))

    async def _resolve_target_balance(self, target: CopyTradingTarget) -> Decimal | None:
        """
        Recupere le capital de reference d'une cible distante.

        Args:
            target (CopyTradingTarget): Cible distante.

        Returns:
            Decimal | None: Capital de reference estime.
        """
        if target.balance_reference is not None:
            return Decimal(str(target.balance_reference))

        try:
            response = await self._http_client.get(
                self._join_url(target.banker_base_url, "/account"),
                headers=get_internal_headers("banker"),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(
                "Copy trading: impossible de lire le compte distant %s: %s",
                target.name,
                exc,
            )
            return None

        if target.use_equity_for_sizing:
            raw_value = payload.get("equity", payload.get("balance", 0))
        else:
            raw_value = payload.get("balance", payload.get("equity", 0))
        try:
            return Decimal(str(raw_value))
        except Exception:
            return None

    def _scale_volume(
        self,
        master_volume: Decimal,
        master_balance: Decimal | None,
        target_balance: Decimal | None,
        allocation_ratio: Decimal,
        symbol: str | None = None,
    ) -> Decimal:
        """
        Calcule un volume proportionnel entre compte maitre et compte fille.

        Args:
            master_volume (Decimal): Volume execute par le maitre.
            master_balance (Decimal | None): Capital du maitre.
            target_balance (Decimal | None): Capital de la cible.
            allocation_ratio (Decimal): Ratio supplementaire de la cible.
            symbol (str | None): Symbole de la transaction pour regles specifiques.

        Returns:
            Decimal: Volume cible arrondi au centieme inferieur.
        """
        ratio = Decimal(str(allocation_ratio or Decimal("1.0")))

        # Division par 5 sur le Gold (XAUUSD) pour ramener le risque pip/volatilité
        # au même niveau que les paires Forex classiques.
        if symbol:
            symbol_upper = str(symbol).strip().upper()
            if "XAUUSD" in symbol_upper or "GOLD" in symbol_upper:
                ratio /= Decimal("5.0")

        if target_balance is not None and master_balance is not None and master_balance > 0:
            ratio *= target_balance / master_balance

        scaled_volume = (Decimal(str(master_volume)) * ratio).quantize(
            Decimal("0.01"),
            rounding=ROUND_FLOOR,
        )
        if scaled_volume < Decimal("0.01"):
            return Decimal("0")
        return scaled_volume

    def _build_copy_comment(self, original_comment: str) -> str:
        """
        Construit un commentaire court marquant un ordre copie.

        Args:
            original_comment (str): Commentaire source.

        Returns:
            str: Commentaire nettoye et borne a 31 caracteres.
        """
        source_comment = str(original_comment or "").strip()
        if source_comment:
            comment = f"COPY {source_comment}"
        else:
            comment = "COPY EVA"
        return comment[:31]

    def _build_repair_order(
        self,
        master_position: Position,
        resolved_symbol: str,
        volume: Decimal,
    ) -> TradeOrder:
        """
        Construit un ordre de rattrapage pour une position maitre deja ouverte.

        Args:
            master_position (Position): Position maitre a recopier.
            resolved_symbol (str): Symbole cible resolu pour le follower.
            volume (Decimal): Volume follower attendu.

        Returns:
            TradeOrder: Ordre distant de rattrapage.
        """
        return TradeOrder(
            symbol=resolved_symbol,
            action=master_position.action,
            volume=volume,
            entry_price=Decimal(str(master_position.current_price)),
            stop_loss_price=master_position.stop_loss,
            take_profit_price=master_position.take_profit,
            source=OrderSource.COPY,
            comment=self._build_copy_comment(f"REPAIR {master_position.symbol}"),
        )

    def _match_remote_position(
        self,
        remote_positions: list[dict[str, Any]],
        used_remote_tickets: set[int],
        resolved_symbol: str,
        master_position: Position,
        expected_volume: Decimal,
    ) -> dict[str, Any] | None:
        """
        Trouve la meilleure correspondance distante pour une position maitre.

        La recherche exige le meme symbole cible et la meme direction. En cas
        de doublons, le score privilegie le volume, puis l'horodatage, puis le
        prix d'entree.

        Args:
            remote_positions (list[dict[str, Any]]): Positions deja ouvertes.
            used_remote_tickets (set[int]): Tickets distants deja rattaches.
            resolved_symbol (str): Symbole cible attendu.
            master_position (Position): Position maitre a faire correspondre.
            expected_volume (Decimal): Volume follower attendu.

        Returns:
            dict[str, Any] | None: Position distante la plus credible, sinon
            `None`.
        """
        matching_candidates: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
        normalized_symbol = self._normalize_symbol_token(resolved_symbol)
        expected_action = (
            master_position.action.value
            if isinstance(master_position.action, TradeAction)
            else str(master_position.action)
        ).upper()
        master_open_time = master_position.open_time
        master_open_price = float(master_position.open_price)

        for remote_position in remote_positions:
            try:
                remote_ticket = int(remote_position.get("ticket"))
            except (TypeError, ValueError):
                continue
            if remote_ticket in used_remote_tickets:
                continue
            remote_symbol = str(remote_position.get("symbol") or "")
            if self._normalize_symbol_token(remote_symbol) != normalized_symbol:
                continue
            remote_action = str(remote_position.get("action") or "").upper()
            if remote_action != expected_action:
                continue

            volume_score = self._compute_volume_distance(
                expected_volume=expected_volume,
                remote_position=remote_position,
            )
            time_score = self._compute_time_distance(
                master_open_time=master_open_time,
                remote_position=remote_position,
            )
            price_score = self._compute_price_distance(
                master_open_price=master_open_price,
                remote_position=remote_position,
            )
            matching_candidates.append(((volume_score, time_score, price_score), remote_position))

        if not matching_candidates:
            return None

        matching_candidates.sort(key=lambda item: item[0])
        return matching_candidates[0][1]

    @staticmethod
    def _compute_volume_distance(
        expected_volume: Decimal,
        remote_position: dict[str, Any],
    ) -> float:
        """
        Calcule l'ecart de volume entre le maitre et un follower.

        Args:
            expected_volume (Decimal): Volume attendu pour la cible.
            remote_position (dict[str, Any]): Position distante candidate.

        Returns:
            float: Ecart absolu de volume.
        """
        try:
            remote_volume = Decimal(str(remote_position.get("volume") or "0"))
        except Exception:
            return float("inf")
        return float(abs(remote_volume - expected_volume))

    @staticmethod
    def _compute_time_distance(
        master_open_time: datetime,
        remote_position: dict[str, Any],
    ) -> float:
        """
        Calcule l'ecart temporel entre une position maitre et un follower.

        Args:
            master_open_time (datetime): Date d'ouverture de la position maitre.
            remote_position (dict[str, Any]): Position distante candidate.

        Returns:
            float: Ecart absolu en secondes.
        """
        raw_open_time = remote_position.get("open_time")
        if not raw_open_time:
            return float("inf")
        try:
            parsed_open_time = datetime.fromisoformat(str(raw_open_time))
        except ValueError:
            return float("inf")
        return abs((parsed_open_time - master_open_time).total_seconds())

    @staticmethod
    def _compute_price_distance(
        master_open_price: float,
        remote_position: dict[str, Any],
    ) -> float:
        """
        Calcule l'ecart de prix d'entree entre maitre et follower.

        Args:
            master_open_price (float): Prix d'entree du maitre.
            remote_position (dict[str, Any]): Position distante candidate.

        Returns:
            float: Ecart absolu de prix.
        """
        try:
            remote_open_price = float(remote_position.get("open_price") or 0.0)
        except (TypeError, ValueError):
            return float("inf")
        return abs(remote_open_price - master_open_price)

    async def _resolve_target_symbol(
        self, target: CopyTradingTarget, source_symbol: str
    ) -> str | None:
        """
        Traduit un symbole source vers le meilleur symbole cible disponible.

        La resolution privilegie d'abord le mapping explicite, puis l'univers
        de symboles renvoye par l'instance cible, puis la liste statique
        `supported_symbols` si elle est fournie.

        Args:
            target (CopyTradingTarget): Cible distante a evaluer.
            source_symbol (str): Symbole d'origine du compte maitre.

        Returns:
            str | None: Symbole cible resolu, ou `None` si aucun candidat
            compatible n'existe.
        """
        normalized_source = self._normalize_symbol_token(source_symbol)
        explicit_map = {
            self._normalize_symbol_token(key): str(value).strip()
            for key, value in target.symbol_map.items()
            if str(value).strip()
        }
        configured_catalog = self._build_symbol_catalog(target.supported_symbols)
        remote_catalog = await self._fetch_target_symbol_catalog(target)
        effective_catalog = remote_catalog or configured_catalog

        candidates: list[str] = []
        explicit_symbol = explicit_map.get(normalized_source)
        if explicit_symbol:
            candidates.extend(self._build_symbol_candidates(explicit_symbol))
        candidates.extend(self._build_symbol_candidates(source_symbol))

        if effective_catalog:
            for candidate in candidates:
                actual_symbol = effective_catalog.get(self._normalize_symbol_token(candidate))
                if actual_symbol:
                    return actual_symbol
            for candidate in candidates:
                actual_symbol = self._resolve_catalog_symbol_by_suffix(
                    effective_catalog=effective_catalog,
                    candidate=candidate,
                )
                if actual_symbol:
                    return actual_symbol
            return None

        return candidates[0] if candidates else None

    async def _fetch_target_symbol_catalog(self, target: CopyTradingTarget) -> dict[str, str]:
        """
        Lit et met en cache l'univers de symboles d'une cible distante.

        Args:
            target (CopyTradingTarget): Cible distante a interroger.

        Returns:
            dict[str, str]: Catalogue `symbole_normalise -> symbole_reel`.
        """
        cached_entry = self._target_symbols_cache.get(target.id)
        now = time.monotonic()
        if cached_entry and now - cached_entry[0] < self._target_symbols_cache_ttl_seconds:
            return cached_entry[1]

        try:
            response = await self._http_client.get(
                self._join_url(target.banker_base_url, "/symbols/discover"),
                headers=get_internal_headers("banker"),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.info(
                "Copy trading: catalogue de symboles distant indisponible pour %s: %s",
                target.name,
                exc,
            )
            return {}

        raw_symbols = payload.get("symbols", []) if isinstance(payload, dict) else []
        if not isinstance(raw_symbols, list):
            return {}

        catalog = self._build_symbol_catalog(raw_symbols)
        self._target_symbols_cache[target.id] = (now, catalog)
        return catalog

    def _build_symbol_catalog(self, symbols: list[str]) -> dict[str, str]:
        """
        Construit un catalogue normalise a partir d'une liste de symboles.

        Args:
            symbols (list[str]): Symboles bruts a indexer.

        Returns:
            dict[str, str]: Dictionnaire `symbole_normalise -> symbole_reel`.
        """
        catalog: dict[str, str] = {}
        for symbol in symbols:
            cleaned = str(symbol or "").strip()
            if not cleaned:
                continue
            catalog[self._normalize_symbol_token(cleaned)] = cleaned
        return catalog

    def _build_symbol_candidates(self, symbol: str) -> list[str]:
        """
        Produit des variantes plausibles pour un symbole inter-brokers.

        Args:
            symbol (str): Symbole source a traduire.

        Returns:
            list[str]: Candidats tries du plus direct au plus permissif.
        """
        cleaned = str(symbol or "").strip()
        if not cleaned:
            return []

        upper_symbol = cleaned.upper()
        candidates = [cleaned, upper_symbol]
        alias_map = {
            "XAUUSD": ["GOLD", "XAUUSD", "XAUUSD."],
            "US100.CASH": ["US100", "NAS100", "USTEC", "NASDAQ100"],
            "US30.CASH": ["US30", "DJ30", "WALLSTREET", "DJIA30"],
            "GER40.CASH": ["GER40", "DE40", "DAX40", "GERMANY40"],
            "US500.CASH": ["US500", "SPX500", "SP500", "US500"],
            "BTCUSD": ["BTCUSD", "XBTUSD"],
        }
        candidates.extend(alias_map.get(upper_symbol, []))

        if upper_symbol.endswith(".CASH"):
            candidates.append(upper_symbol.replace(".CASH", ""))
        if "." in upper_symbol:
            candidates.append(upper_symbol.replace(".", ""))
        candidates.extend([f"{upper_symbol}.E", f"{upper_symbol}.M"])

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = self._normalize_symbol_token(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _normalize_symbol_token(symbol: str) -> str:
        """
        Normalise un symbole pour la comparaison inter-brokers.

        Args:
            symbol (str): Symbole brut.

        Returns:
            str: Forme majuscule sans ponctuation sensible.
        """
        return (
            str(symbol or "")
            .strip()
            .upper()
            .replace("_", "")
            .replace("-", "")
            .replace(".", "")
            .replace("/", "")
            .replace(" ", "")
        )

    @staticmethod
    def _resolve_catalog_symbol_by_suffix(
        effective_catalog: dict[str, str],
        candidate: str,
    ) -> str | None:
        """
        Tente une resolution permissive pour les brokers a suffixe.

        Cette passe sert surtout pour des univers comme FTUK qui exposent des
        symboles du type `EURUSD.e` ou `XAUUSD.m` alors que le compte maitre
        travaille avec `EURUSD` et `XAUUSD`.

        Args:
            effective_catalog (dict[str, str]): Catalogue normalise de la cible.
            candidate (str): Symbole candidat issu du compte maitre.

        Returns:
            str | None: Symbole reel de la cible si une correspondance
            raisonnable est trouvee, sinon `None`.
        """
        normalized_candidate = CopyTradingRouter._normalize_symbol_token(candidate)
        if not normalized_candidate:
            return None

        prefix_matches = [
            actual_symbol
            for normalized_symbol, actual_symbol in effective_catalog.items()
            if normalized_symbol.startswith(normalized_candidate)
        ]
        if not prefix_matches:
            return None

        prefix_matches.sort(key=lambda value: (len(value), value))
        return prefix_matches[0]

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        """
        Assemble une URL de base et un chemin HTTP.

        Args:
            base_url (str): URL de base de l'instance cible.
            path (str): Chemin HTTP a joindre.

        Returns:
            str: URL absolue assemblee.
        """
        return f"{str(base_url).rstrip('/')}{path}"
