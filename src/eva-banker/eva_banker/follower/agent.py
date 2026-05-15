"""Moteur local de l'agent follower distribue."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Callable

from eva_banker.follower.config import FollowerAgentConfig
from eva_banker.follower.models import (
    FollowerCommand,
    FollowerCommandType,
    FollowerExecutionResult,
    FollowerRuntimeStatus,
)
from eva_banker.follower.relay import RelayClient
from eva_banker.services.mt5 import MT5Service
from shared.models import OrderSource, Position, TradeOrder

logger = logging.getLogger(__name__)

EventCallback = Callable[[str], None]


class FollowerAgent:
    """Agent local qui execute les commandes de copy trading sur MT5."""

    def __init__(
        self,
        config: FollowerAgentConfig,
        *,
        mt5_service: Any | None = None,
        relay_client: Any | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        """Initialise l'agent follower.

        Args:
            config (FollowerAgentConfig): Configuration locale.
            mt5_service (Any | None): Service MT5 injecte pour tests.
            relay_client (Any | None): Client relay injecte pour tests.
            event_callback (EventCallback | None): Callback UI/log.
        """

        self.config = config
        self.mt5_service = mt5_service or MT5Service(
            mock_mode=config.mock_mt5,
            login=config.mt5_login,
            password=config.mt5_password,
            server=config.mt5_server,
            terminal_path=config.mt5_terminal_path,
            terminal_portable=config.mt5_terminal_portable,
        )
        self.relay_client = relay_client or RelayClient(config)
        self.event_callback = event_callback
        self.running = False
        self.paused = False
        self.relay_connected = False
        self.last_error: str | None = None
        self.last_event: str | None = None
        self._last_heartbeat_at = 0.0
        self._processed_commands: set[str] = set()
        self._ticket_links: dict[str, int] = {}
        self._last_account_reference: Decimal | None = None
        self._load_state()

    async def close(self) -> None:
        """Ferme les connexions de l'agent."""

        self.running = False
        if hasattr(self.relay_client, "close"):
            await self.relay_client.close()
        if hasattr(self.mt5_service, "disconnect"):
            await self.mt5_service.disconnect()

    async def connect_mt5(self) -> bool:
        """Connecte le terminal MT5 local.

        Returns:
            bool: True si le terminal est pret.
        """

        if self.config.dry_run:
            self._emit("Mode dry-run actif: connexion MT5 non requise.")
            return True
        connected = await self.mt5_service.connect()
        if connected:
            await self._refresh_local_account_reference()
            self._emit("Connexion MT5 follower etablie.")
        else:
            self._emit("Connexion MT5 follower impossible.")
        return bool(connected)

    async def run_forever(self) -> None:
        """Demarre la boucle principale de polling relay."""

        self.running = True
        await self.connect_mt5()
        self._emit("Agent follower demarre.")
        while self.running:
            try:
                await self.run_once()
            except Exception as exc:
                self.last_error = str(exc)
                self._emit(f"Erreur boucle follower: {exc}")
            await asyncio.sleep(max(0.2, float(self.config.poll_interval_seconds)))

    async def run_once(self) -> list[FollowerExecutionResult]:
        """Execute un cycle de lecture relay puis heartbeat.

        Returns:
            list[FollowerExecutionResult]: Resultats produits pendant le cycle.
        """

        if self.paused:
            await self._maybe_send_heartbeat()
            return []

        commands = await self.relay_client.fetch_commands(after=self._last_processed_command())
        self.relay_connected = True
        results: list[FollowerExecutionResult] = []
        for command in commands:
            result = await self.handle_command(command)
            results.append(result)
            try:
                await self.relay_client.acknowledge(result)
            except Exception as exc:
                self._emit(f"Ack relay impossible pour {command.command_id}: {exc}")
        await self._maybe_send_heartbeat()
        return results

    async def handle_command(self, command: FollowerCommand) -> FollowerExecutionResult:
        """Execute une commande idempotente du relay.

        Args:
            command (FollowerCommand): Commande recue.

        Returns:
            FollowerExecutionResult: Resultat local.
        """

        if command.command_id in self._processed_commands:
            return FollowerExecutionResult(
                command_id=command.command_id,
                success=True,
                message="Commande deja traitee.",
                master_ticket=command.master_ticket,
                mode="idempotent_skip",
            )

        if command.command_type == FollowerCommandType.OPEN:
            result = await self._execute_open(command)
        elif command.command_type == FollowerCommandType.CLOSE:
            result = await self._execute_close(command)
        elif command.command_type == FollowerCommandType.MODIFY:
            result = await self._execute_modify(command)
        elif command.command_type in {FollowerCommandType.SYNC, FollowerCommandType.PING}:
            result = FollowerExecutionResult(
                command_id=command.command_id,
                success=True,
                message="Commande de synchronisation recue.",
                master_ticket=command.master_ticket,
                mode=command.command_type.value,
            )
        else:
            result = FollowerExecutionResult(
                command_id=command.command_id,
                success=False,
                message=f"Type de commande inconnu: {command.command_type}",
                master_ticket=command.master_ticket,
                mode="unsupported",
            )

        if result.success:
            self._processed_commands.add(command.command_id)
            self._save_state()
        self._emit(f"{command.command_type.value}: {result.message}")
        return result

    def get_status(self) -> FollowerRuntimeStatus:
        """Retourne l'etat courant de l'agent.

        Returns:
            FollowerRuntimeStatus: Snapshot runtime.
        """

        return FollowerRuntimeStatus(
            client_id=self.config.client_id,
            account_label=self.config.account_label,
            relay_connected=self.relay_connected,
            mt5_connected=bool(getattr(self.mt5_service, "is_connected", False)) or self.config.dry_run,
            running=self.running,
            paused=self.paused,
            dry_run=self.config.dry_run,
            processed_commands=len(self._processed_commands),
            linked_positions=len(self._ticket_links),
            last_error=self.last_error,
            last_event=self.last_event,
        )

    async def _execute_open(self, command: FollowerCommand) -> FollowerExecutionResult:
        """Ouvre une position locale depuis une commande maitre."""

        if not command.symbol or command.action is None or command.volume is None:
            return self._failure(command, "Commande d'ouverture incomplete.", mode="open")

        local_symbol = self._resolve_symbol(command.symbol)
        if not local_symbol:
            return self._failure(
                command,
                f"Symbole non supporte localement: {command.symbol}",
                mode="open",
            )
        volume = await self._scale_volume(command.volume, command)
        if volume <= Decimal("0"):
            return self._failure(command, "Volume local nul apres scaling.", mode="open")

        if self.config.dry_run:
            ticket = self._dry_ticket(command)
            self._link_ticket(command, ticket)
            return FollowerExecutionResult(
                command_id=command.command_id,
                success=True,
                message=f"[dry-run] Ouverture simulee {local_symbol} {volume}.",
                ticket=ticket,
                master_ticket=command.master_ticket,
                local_symbol=local_symbol,
                mode="dry_open",
            )

        order = TradeOrder(
            symbol=local_symbol,
            action=command.action,
            volume=volume,
            entry_price=command.entry_price,
            stop_loss_price=command.stop_loss,
            take_profit_price=command.take_profit,
            source=OrderSource.COPY,
            comment=self._build_comment(command),
        )
        result = await self.mt5_service.execute_order(order)
        if not result.get("success"):
            return self._failure(command, str(result.get("message") or "Ouverture refusee."), mode="open")
        ticket = int(result.get("ticket") or 0)
        self._link_ticket(command, ticket)
        return FollowerExecutionResult(
            command_id=command.command_id,
            success=True,
            message=str(result.get("message") or "Ouverture executee."),
            ticket=ticket,
            master_ticket=command.master_ticket,
            local_symbol=local_symbol,
            mode="open",
            details=dict(result),
        )

    async def _execute_close(self, command: FollowerCommand) -> FollowerExecutionResult:
        """Ferme ou transforme une position locale en runner protege."""

        ticket = self._resolve_linked_ticket(command)
        if not ticket:
            return self._failure(command, "Ticket local introuvable pour la cloture.", mode="close")

        close_reason = str(command.close_reason or "").upper()
        profitable = Decimal(str(command.master_profit or "0")) > Decimal("0")
        keep_runner = profitable and not command.full_close and close_reason not in {"SL", "STOP_LOSS", "LOSS"}

        if self.config.dry_run:
            mode = "dry_runner" if keep_runner else "dry_full_close"
            if not keep_runner:
                self._unlink_ticket(command)
            return FollowerExecutionResult(
                command_id=command.command_id,
                success=True,
                message=f"[dry-run] Cloture simulee ticket {ticket}.",
                ticket=ticket,
                master_ticket=command.master_ticket,
                mode=mode,
            )

        if keep_runner:
            return await self._close_profit_runner(command, ticket)

        result = await self.mt5_service.close_position(ticket)
        if result.get("success"):
            self._unlink_ticket(command)
        return FollowerExecutionResult(
            command_id=command.command_id,
            success=bool(result.get("success")),
            message=str(result.get("message") or "Cloture totale executee."),
            ticket=ticket,
            master_ticket=command.master_ticket,
            mode="full_close",
            details=dict(result),
        )

    async def _execute_modify(self, command: FollowerCommand) -> FollowerExecutionResult:
        """Modifie le SL/TP d'une position locale liee."""

        ticket = self._resolve_linked_ticket(command)
        if not ticket:
            return self._failure(command, "Ticket local introuvable pour modification.", mode="modify")
        if self.config.dry_run:
            return FollowerExecutionResult(
                command_id=command.command_id,
                success=True,
                message=f"[dry-run] Modification simulee ticket {ticket}.",
                ticket=ticket,
                master_ticket=command.master_ticket,
                mode="dry_modify",
            )
        result = await self.mt5_service.modify_position(
            ticket,
            sl=float(command.stop_loss or 0),
            tp=float(command.take_profit or 0),
        )
        return FollowerExecutionResult(
            command_id=command.command_id,
            success=bool(result.get("success")),
            message=str(result.get("message") or "Modification executee."),
            ticket=ticket,
            master_ticket=command.master_ticket,
            mode="modify",
            details=dict(result),
        )

    async def _close_profit_runner(
        self,
        command: FollowerCommand,
        ticket: int,
    ) -> FollowerExecutionResult:
        """Ferme 70% d'une position gagnante puis remonte le SL au BE."""

        position = await self._find_position(ticket)
        if position is None:
            return self._failure(command, "Position locale absente pour runner.", mode="runner")

        current_volume = Decimal(str(position.volume or "0"))
        close_volume = (current_volume * Decimal("0.70")).quantize(
            Decimal("0.01"),
            rounding=ROUND_FLOOR,
        )
        remaining_volume = current_volume - close_volume
        if close_volume <= Decimal("0") or remaining_volume < Decimal("0.01"):
            result = await self.mt5_service.close_position(ticket)
            if result.get("success"):
                self._unlink_ticket(command)
            return FollowerExecutionResult(
                command_id=command.command_id,
                success=bool(result.get("success")),
                message="Volume trop petit pour runner, cloture totale appliquee.",
                ticket=ticket,
                master_ticket=command.master_ticket,
                mode="runner_full_close_fallback",
                details=dict(result),
            )

        close_result = await self.mt5_service.close_position(ticket, volume=close_volume)
        if not close_result.get("success"):
            return FollowerExecutionResult(
                command_id=command.command_id,
                success=False,
                message=str(close_result.get("message") or "Cloture partielle refusee."),
                ticket=ticket,
                master_ticket=command.master_ticket,
                mode="runner_partial_failed",
                details=dict(close_result),
            )

        break_even = float(position.open_price or 0)
        modify_result = await self.mt5_service.modify_position(ticket, sl=break_even, tp=0.0)
        if not modify_result.get("success"):
            fallback = await self.mt5_service.close_position(ticket)
            if fallback.get("success"):
                self._unlink_ticket(command)
            return FollowerExecutionResult(
                command_id=command.command_id,
                success=bool(fallback.get("success")),
                message="SLBE impossible, reliquat ferme par securite.",
                ticket=ticket,
                master_ticket=command.master_ticket,
                mode="runner_slbe_failed_full_close",
                details={"partial": close_result, "modify": modify_result, "fallback": fallback},
            )

        return FollowerExecutionResult(
            command_id=command.command_id,
            success=True,
            message="Runner protege: 70% ferme et SL au break-even.",
            ticket=ticket,
            master_ticket=command.master_ticket,
            mode="runner_70_slbe",
            details={"partial": close_result, "modify": modify_result, "break_even": break_even},
        )

    async def _find_position(self, ticket: int) -> Position | None:
        """Retrouve une position locale par ticket."""

        positions = await self.mt5_service.get_open_positions()
        if not positions:
            return None
        for position in positions:
            if int(position.ticket) == int(ticket):
                return position
        return None

    def _resolve_symbol(self, source_symbol: str) -> str | None:
        """Traduit un symbole maitre vers le symbole broker local."""

        direct = str(self.config.symbol_map.get(source_symbol, "") or "").strip()
        local_symbol = direct or source_symbol
        if not self.config.supported_symbols:
            return local_symbol
        normalized_allowed = {self._normalize_symbol(item): item for item in self.config.supported_symbols}
        return normalized_allowed.get(self._normalize_symbol(local_symbol))

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalise un symbole pour comparaison broker."""

        return "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())

    async def _scale_volume(self, volume: Decimal, command: FollowerCommand) -> Decimal:
        """Calcule le volume dynamique puis normalise au pas centieme.

        Args:
            volume (Decimal): Volume execute sur le compte maitre.
            command (FollowerCommand): Commande source contenant les metadonnees maitre.

        Returns:
            Decimal: Volume follower arrondi au centieme inferieur.
        """

        ratio = self._resolve_risk_multiplier()
        master_reference = self._resolve_master_reference(command)
        local_reference = await self._resolve_local_reference()
        if master_reference is not None and local_reference is not None and master_reference > 0:
            ratio *= local_reference / master_reference

        raw_volume = Decimal(str(volume)) * ratio
        return raw_volume.quantize(Decimal("0.01"), rounding=ROUND_FLOOR)

    def _resolve_risk_multiplier(self) -> Decimal:
        """Retourne le multiplicateur de risque manuel.

        Returns:
            Decimal: Multiplicateur de risque, 1.0 par defaut.
        """

        try:
            return Decimal(str(self.config.allocation_ratio or 1.0))
        except Exception:
            return Decimal("1.0")

    def _resolve_master_reference(self, command: FollowerCommand) -> Decimal | None:
        """Recupere le capital de reference du maitre.

        Args:
            command (FollowerCommand): Commande source.

        Returns:
            Decimal | None: Capital maitre disponible.
        """

        payload = command.payload or {}
        for key in ("master_equity", "master_balance", "master_balance_reference"):
            raw_value = payload.get(key)
            if raw_value not in (None, ""):
                try:
                    return Decimal(str(raw_value))
                except Exception:
                    continue
        if self.config.master_balance_reference is None:
            return None
        try:
            return Decimal(str(self.config.master_balance_reference))
        except Exception:
            return None

    async def _resolve_local_reference(self) -> Decimal | None:
        """Recupere le capital local utilise pour le sizing dynamique.

        Returns:
            Decimal | None: Capital follower disponible.
        """

        if self.config.balance_reference is not None:
            try:
                return Decimal(str(self.config.balance_reference))
            except Exception:
                return None
        if self._last_account_reference is not None:
            return self._last_account_reference
        return await self._refresh_local_account_reference()

    async def _refresh_local_account_reference(self) -> Decimal | None:
        """Met en cache l'equity ou la balance locale MT5.

        Returns:
            Decimal | None: Capital local lu depuis MT5.
        """

        if self.config.dry_run or not hasattr(self.mt5_service, "get_account_info"):
            return None
        try:
            account = await self.mt5_service.get_account_info()
        except Exception as exc:
            self._emit(f"Lecture compte follower impossible pour sizing: {exc}")
            return None
        if account is None:
            return None
        raw_value = account.equity if self.config.use_equity_for_sizing else account.balance
        try:
            self._last_account_reference = Decimal(str(raw_value))
        except Exception:
            self._last_account_reference = None
        return self._last_account_reference

    def _resolve_linked_ticket(self, command: FollowerCommand) -> int | None:
        """Retourne le ticket local associe a une commande maitre."""

        if command.master_ticket is None:
            raw_ticket = command.payload.get("local_ticket")
            return int(raw_ticket) if raw_ticket else None
        return self._ticket_links.get(str(command.master_ticket))

    def _link_ticket(self, command: FollowerCommand, ticket: int) -> None:
        """Relie un ticket maitre a un ticket local."""

        if command.master_ticket is not None and ticket:
            self._ticket_links[str(command.master_ticket)] = int(ticket)

    def _unlink_ticket(self, command: FollowerCommand) -> None:
        """Supprime un lien ticket apres cloture totale."""

        if command.master_ticket is not None:
            self._ticket_links.pop(str(command.master_ticket), None)

    def _failure(self, command: FollowerCommand, message: str, *, mode: str) -> FollowerExecutionResult:
        """Construit un resultat d'echec homogene."""

        return FollowerExecutionResult(
            command_id=command.command_id,
            success=False,
            message=message,
            master_ticket=command.master_ticket,
            mode=mode,
        )

    def _build_comment(self, command: FollowerCommand) -> str:
        """Construit un commentaire MT5 compact et audit able."""

        suffix = f"M{command.master_ticket}" if command.master_ticket is not None else command.command_id[:10]
        raw = command.comment or "COPY HIVE"
        return f"{raw[:18]} {suffix}"[:31]

    @staticmethod
    def _dry_ticket(command: FollowerCommand) -> int:
        """Cree un ticket stable pour le mode dry-run."""

        base = abs(hash(command.command_id)) % 900000
        return 90000000 + base

    def _last_processed_command(self) -> str | None:
        """Retourne une borne de polling approximative pour le relay."""

        if not self._processed_commands:
            return None
        return sorted(self._processed_commands)[-1]

    async def _maybe_send_heartbeat(self) -> None:
        """Envoie un heartbeat si l'intervalle configure est depasse."""

        now = time.time()
        if now - self._last_heartbeat_at < max(1.0, self.config.heartbeat_interval_seconds):
            return
        self._last_heartbeat_at = now
        status = self.get_status()
        payload = status.model_dump(mode="json") if hasattr(status, "model_dump") else status.dict()
        try:
            await self.relay_client.heartbeat(payload)
        except Exception as exc:
            self.relay_connected = False
            self.last_error = str(exc)
            self._emit(f"Heartbeat relay impossible: {exc}")

    def _load_state(self) -> None:
        """Charge les liens locaux et commandes deja traitees."""

        state_path = Path(self.config.state_path)
        if not state_path.exists():
            return
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.last_error = f"Etat follower illisible: {exc}"
            return
        self._processed_commands = set(str(item) for item in payload.get("processed_commands", []))
        self._ticket_links = {
            str(key): int(value)
            for key, value in dict(payload.get("ticket_links", {})).items()
            if value
        }

    def _save_state(self) -> None:
        """Sauvegarde l'etat minimal d'idempotence."""

        state_path = Path(self.config.state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(),
            "processed_commands": sorted(self._processed_commands)[-1000:],
            "ticket_links": self._ticket_links,
        }
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _emit(self, message: str) -> None:
        """Publie un evenement dans les logs et l'interface."""

        self.last_event = message
        logger.info(message)
        log_path = Path(self.config.log_path)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
        except OSError:
            pass
        if self.event_callback:
            self.event_callback(message)
