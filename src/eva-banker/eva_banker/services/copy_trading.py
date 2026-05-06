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
import time
from decimal import Decimal, ROUND_FLOOR
from typing import Any
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from shared import OrderSource, TradeAction, TradeOrder, get_settings
from shared.internal_auth import get_internal_headers

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
        self._http_client = httpx.AsyncClient(
            timeout=max(2.0, float(self.settings.banker_copy_request_timeout_seconds))
        )
        self._target_symbols_cache_ttl_seconds = 300.0

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

    async def close_position(self, ticket: int) -> dict[str, Any]:
        """
        Ferme la position maitre puis tente de fermer les positions filles.

        Args:
            ticket (int): Ticket maitre local.

        Returns:
            dict[str, Any]: Resultat local enrichi des retours de replication.
        """
        local_result = await self.primary_service.close_position(ticket)
        if not local_result.get("success"):
            return local_result

        remote_results = await self._close_remote_links(
            ticket,
            close_as_runner=self._should_keep_remote_runner(local_result),
        )
        if remote_results:
            local_result["copy_results"] = remote_results
        return local_result

    async def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> dict[str, Any]:
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
            self._execute_scaled_copy(target=target, master_order=master_order, master_balance=master_balance)
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
            "action": order.action.value if isinstance(order.action, TradeAction) else str(order.action),
            "volume": str(order.volume),
            "stop_loss": str(order.stop_loss_price) if order.stop_loss_price is not None else None,
            "take_profit": str(order.take_profit_price) if order.take_profit_price is not None else None,
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
            return {
                "target_id": str(target.id),
                "target_name": target.name,
                "success": False,
                "message": f"Execution distante impossible: {exc}",
            }

        data["target_id"] = str(target.id)
        data["target_name"] = target.name
        return data

    def _should_keep_remote_runner(self, local_result: dict[str, Any]) -> bool:
        """
        Indique si une cloture gagnante du maitre doit devenir un runner distant.

        Args:
            local_result (dict[str, Any]): Resultat de cloture du maitre.

        Returns:
            bool: True si les followers doivent garder une demi-position au BE.
        """
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
                normalized.append({"success": False, "message": f"Echec cloture distante: {result}"})
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
            dict[str, Any]: Resultat composite de la demi-cloture et du passage au BE.
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

        partial_result = await self._close_remote_ticket(
            target,
            remote_ticket,
            volume=current_volume / Decimal("2"),
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
        partial_result["runner_mode"] = "half_close_break_even"
        partial_result["break_even_price"] = break_even_price
        partial_result["modify_result"] = modify_result
        partial_result["success"] = bool(partial_result.get("success")) and bool(
            modify_result.get("success")
        )
        if not modify_result.get("success"):
            partial_result["message"] = (
                "Demi-cloture distante reussie, mais passage au break-even echoue."
            )
        return partial_result

    async def _modify_remote_links(self, master_ticket: int, sl: float, tp: float) -> list[dict[str, Any]]:
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
                normalized.append({"success": False, "message": f"Echec modification distante: {result}"})
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
    ) -> Decimal:
        """
        Calcule un volume proportionnel entre compte maitre et compte fille.

        Args:
            master_volume (Decimal): Volume execute par le maitre.
            master_balance (Decimal | None): Capital du maitre.
            target_balance (Decimal | None): Capital de la cible.
            allocation_ratio (Decimal): Ratio supplementaire de la cible.

        Returns:
            Decimal: Volume cible arrondi au centieme inferieur.
        """
        ratio = Decimal(str(allocation_ratio or Decimal("1.0")))
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

    async def _resolve_target_symbol(self, target: CopyTradingTarget, source_symbol: str) -> str | None:
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
