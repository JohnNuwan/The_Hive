"""Tests `unittest` du routeur de copy trading multi-instances."""

import asyncio
import unittest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

from eva_banker.services.copy_trading import (
    CopyTradingRouter,
    CopyTradingTarget,
    RemoteTicketLink,
)
from shared import AccountBalance, OrderSource, Position, TradeAction, TradeOrder


class DummyPrimaryService:
    """Double simple du service MT5 principal pour les tests."""

    def __init__(self) -> None:
        self.is_connected = True
        self.mock_mode = True
        self.executed_orders: list[TradeOrder] = []
        self.closed_tickets: list[int] = []
        self.modified_tickets: list[tuple[int, float, float]] = []
        self.open_positions: list[Position] = []

    async def execute_order(self, order: TradeOrder) -> dict[str, object]:
        """Memorise l'ordre local et retourne un ticket stable."""
        self.executed_orders.append(order)
        return {
            "success": True,
            "ticket": 111,
            "message": "Ordre local execute.",
        }

    async def close_position(self, ticket: int, volume: Decimal | None = None) -> dict[str, object]:
        """Memorise la cloture locale."""
        self.closed_tickets.append(ticket)
        return {
            "success": True,
            "ticket": ticket,
            "message": "Position locale fermee.",
            "profit": 0.0,
        }

    async def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> dict[str, object]:
        """Memorise la modification locale."""
        self.modified_tickets.append((ticket, sl, tp))
        return {"success": True, "ticket": ticket, "message": "Position locale modifiee."}

    async def get_account_info(self) -> AccountBalance:
        """Retourne un capital fixe pour le compte maitre."""
        return AccountBalance(
            login=123456,
            server="Mock-Server",
            balance=Decimal("100000"),
            equity=Decimal("100000"),
            margin=Decimal("0"),
            free_margin=Decimal("100000"),
        )

    async def disconnect(self) -> None:
        """Rien a faire pour le double de test."""

    async def get_open_positions(self) -> list[Position]:
        """Retourne les positions ouvertes configurees pour le test."""
        return list(self.open_positions)


class CopyTradingRouterTests(unittest.TestCase):
    """Verifie le routage de copie proportionnelle et le garde-fou anti-boucle."""

    def test_scale_volume_uses_balance_ratio(self) -> None:
        """Verifie la proportionalite du volume selon les capitaux."""
        router = CopyTradingRouter(DummyPrimaryService())
        scaled_volume = router._scale_volume(
            master_volume=Decimal("1.00"),
            master_balance=Decimal("100000"),
            target_balance=Decimal("50000"),
            allocation_ratio=Decimal("1.0"),
        )

        self.assertEqual(scaled_volume, Decimal("0.50"))
        asyncio.run(router.close())

    def test_execute_order_fans_out_and_closes_remote_links(self) -> None:
        """Verifie l'ouverture puis la cloture des tickets copies."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 2",
            banker_base_url="http://banker-2:8100",
            allocation_ratio=Decimal("1.0"),
        )
        router.targets[target.id] = target

        router._resolve_target_balance = AsyncMock(return_value=Decimal("50000"))
        router._execute_remote_target_order = AsyncMock(
            return_value={
                "success": True,
                "ticket": 222,
                "message": "Ordre copie execute.",
            }
        )
        router._close_remote_ticket = AsyncMock(
            return_value={
                "success": True,
                "ticket": 222,
                "message": "Ticket copie ferme.",
            }
        )

        order = TradeOrder(
            symbol="XAUUSD",
            action=TradeAction.BUY,
            volume=Decimal("1.00"),
            stop_loss_price=Decimal("4800"),
        )

        local_result = asyncio.run(router.execute_order(order))
        close_result = asyncio.run(router.close_position(111))
        asyncio.run(router.close())

        self.assertEqual(len(primary.executed_orders), 1)
        self.assertEqual(local_result["copy_summary"]["success"], 1)
        self.assertEqual(local_result["copy_results"][0]["scaled_volume"], 0.5)
        self.assertEqual(close_result["copy_results"][0]["ticket"], 222)

    def test_close_position_transmet_le_volume_local(self) -> None:
        """Verifie qu'une cloture partielle transmet bien le volume au service primaire."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        primary.close_position = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "success": True,
                "ticket": 111,
                "message": "Position locale partiellement fermee.",
                "profit": 5.0,
            }
        )

        result = asyncio.run(router.close_position(111, volume=Decimal("0.25")))
        asyncio.run(router.close())

        primary.close_position.assert_awaited_once_with(111, volume=Decimal("0.25"))
        self.assertTrue(result["success"])

    def test_profitable_close_keeps_partial_runner_on_followers(self) -> None:
        """Verifie la cloture 70%% follower puis le passage du stop au BE."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 50K",
            banker_base_url="http://banker-50k:8110",
        )
        router.targets[target.id] = target
        router._ticket_links[111] = [
            RemoteTicketLink(target_id=target.id, remote_ticket=222)
        ]

        async def profitable_close(_ticket: int, volume: Decimal | None = None) -> dict[str, object]:
            return {
                "success": True,
                "ticket": 111,
                "message": "Position locale fermee.",
                "profit": 42.0,
            }

        primary.close_position = profitable_close  # type: ignore[method-assign]
        router._fetch_remote_position_snapshot = AsyncMock(
            return_value={"ticket": 222, "volume": 0.20, "open_price": 4500.5}
        )
        router._close_remote_ticket = AsyncMock(
            return_value={
                "success": True,
                "ticket": 222,
                "volume_closed": 0.14,
                "volume_remaining": 0.06,
                "message": "Cloture partielle distante reussie.",
            }
        )
        router._modify_remote_ticket = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "BE applique."}
        )

        result = asyncio.run(router.close_position(111))
        asyncio.run(router.close())

        router._close_remote_ticket.assert_awaited_once()
        close_call = router._close_remote_ticket.await_args
        self.assertEqual(close_call.args[0], target)
        self.assertEqual(close_call.args[1], 222)
        self.assertEqual(close_call.kwargs["volume"], Decimal("0.14"))
        router._modify_remote_ticket.assert_awaited_once_with(
            target,
            222,
            sl=4500.5,
            tp=0.0,
        )
        self.assertTrue(result["copy_results"][0]["success"])
        self.assertEqual(result["copy_results"][0]["runner_mode"], "partial_close_break_even")

    def test_profitable_close_keeps_partial_runner_on_master_and_followers(self) -> None:
        """Verifie le split local gagnant puis la propagation runner aux followers."""
        primary = DummyPrimaryService()
        primary.open_positions = [
            Position(
                ticket=111,
                symbol="XAUUSD",
                action=TradeAction.BUY,
                volume=Decimal("0.05"),
                open_price=Decimal("4700.0"),
                current_price=Decimal("4715.0"),
                stop_loss=Decimal("4650.0"),
                take_profit=None,
                profit=Decimal("15.0"),
                open_time=datetime(2026, 5, 8, 9, 0, 0),
            )
        ]
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 50K",
            banker_base_url="http://banker-50k:8110",
        )
        router.targets[target.id] = target
        router._ticket_links[111] = [
            RemoteTicketLink(target_id=target.id, remote_ticket=222)
        ]

        primary.close_position = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "success": True,
                "ticket": 111,
                "message": "Position locale partiellement fermee.",
                "profit": 15.0,
                "volume_closed": 0.03,
                "volume_remaining": 0.02,
                "partial_close": True,
            }
        )
        primary.modify_position = AsyncMock(  # type: ignore[method-assign]
            return_value={"success": True, "ticket": 111, "message": "BE local applique."}
        )
        router._fetch_remote_position_snapshot = AsyncMock(
            return_value={"ticket": 222, "volume": 0.20, "open_price": 4500.5}
        )
        router._close_remote_ticket = AsyncMock(
            return_value={
                "success": True,
                "ticket": 222,
                "volume_closed": 0.14,
                "volume_remaining": 0.06,
                "message": "Cloture partielle distante reussie.",
            }
        )
        router._modify_remote_ticket = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "BE distant applique."}
        )

        result = asyncio.run(router.close_position(111))
        asyncio.run(router.close())

        primary.close_position.assert_awaited_once_with(111, volume=Decimal("0.03"))
        primary.modify_position.assert_awaited_once_with(111, sl=4700.0, tp=0.0)
        self.assertEqual(result["runner_mode"], "partial_close_break_even")
        self.assertEqual(result["copy_close_mode"], "runner")
        self.assertEqual(result["copy_results"][0]["runner_mode"], "partial_close_break_even")

    def test_profitable_close_falls_back_to_full_close_when_local_be_fails(self) -> None:
        """Verifie le repli en cloture totale si le BE local echoue."""
        primary = DummyPrimaryService()
        primary.open_positions = [
            Position(
                ticket=111,
                symbol="XAUUSD",
                action=TradeAction.BUY,
                volume=Decimal("0.05"),
                open_price=Decimal("4700.0"),
                current_price=Decimal("4715.0"),
                stop_loss=Decimal("4650.0"),
                take_profit=None,
                profit=Decimal("15.0"),
                open_time=datetime(2026, 5, 8, 9, 0, 0),
            )
        ]
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 50K",
            banker_base_url="http://banker-50k:8110",
        )
        router.targets[target.id] = target
        router._ticket_links[111] = [
            RemoteTicketLink(target_id=target.id, remote_ticket=222)
        ]

        primary.close_position = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                {
                    "success": True,
                    "ticket": 111,
                    "message": "Position locale partiellement fermee.",
                    "profit": 15.0,
                    "volume_closed": 0.03,
                    "volume_remaining": 0.02,
                    "partial_close": True,
                },
                {
                    "success": True,
                    "ticket": 111,
                    "message": "Reliquat local ferme.",
                    "profit": 0.0,
                    "volume_closed": 0.02,
                    "volume_remaining": 0.0,
                    "partial_close": False,
                },
            ]
        )
        primary.modify_position = AsyncMock(  # type: ignore[method-assign]
            return_value={"success": False, "ticket": 111, "message": "BE local refuse."}
        )
        router._close_remote_ticket = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "Cloture distante complete."}
        )

        result = asyncio.run(router.close_position(111))
        asyncio.run(router.close())

        self.assertEqual(primary.close_position.await_count, 2)
        primary.modify_position.assert_awaited_once_with(111, sl=4700.0, tp=0.0)
        router._close_remote_ticket.assert_awaited_once_with(target, 222)
        self.assertEqual(result["copy_close_mode"], "full_close")
        self.assertEqual(result["runner_mode"], "break_even_full_close_fallback")

    def test_losing_close_fully_closes_followers(self) -> None:
        """Verifie qu'une cloture non profitable ferme totalement les followers."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 50K",
            banker_base_url="http://banker-50k:8110",
        )
        router.targets[target.id] = target
        router._ticket_links[111] = [
            RemoteTicketLink(target_id=target.id, remote_ticket=222)
        ]

        async def losing_close(_ticket: int, volume: Decimal | None = None) -> dict[str, object]:
            return {
                "success": True,
                "ticket": 111,
                "message": "Position locale fermee.",
                "profit": -5.0,
            }

        primary.close_position = losing_close  # type: ignore[method-assign]
        router._close_remote_ticket = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "Cloture distante complete."}
        )
        router._modify_remote_ticket = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "Ne doit pas etre appele."}
        )

        result = asyncio.run(router.close_position(111))
        asyncio.run(router.close())

        router._close_remote_ticket.assert_awaited_once_with(target, 222)
        self.assertEqual(router._modify_remote_ticket.await_count, 0)
        self.assertTrue(result["copy_results"][0]["success"])

    def test_synchronize_external_close_keeps_runner_on_profitable_close(self) -> None:
        """Verifie la synchro follower quand le maitre a deja ete ferme en gain."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 50K",
            banker_base_url="http://banker-50k:8110",
        )
        router.targets[target.id] = target
        router._ticket_links[111] = [
            RemoteTicketLink(target_id=target.id, remote_ticket=222)
        ]
        router._fetch_remote_position_snapshot = AsyncMock(
            return_value={"ticket": 222, "volume": 0.20, "open_price": 4500.5}
        )
        router._close_remote_ticket = AsyncMock(
            return_value={
                "success": True,
                "ticket": 222,
                "volume_closed": 0.14,
                "volume_remaining": 0.06,
                "message": "Cloture partielle distante reussie.",
            }
        )
        router._modify_remote_ticket = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "BE distant applique."}
        )

        result = asyncio.run(router.synchronize_external_close(111, profit=12.0))
        asyncio.run(router.close())

        self.assertTrue(result["success"])
        self.assertEqual(result["copy_summary"]["success"], 1)
        self.assertEqual(result["copy_results"][0]["runner_mode"], "partial_close_break_even")

    def test_synchronize_external_close_fully_closes_losing_followers(self) -> None:
        """Verifie la synchro follower quand le maitre a deja ete ferme en perte."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 50K",
            banker_base_url="http://banker-50k:8110",
        )
        router.targets[target.id] = target
        router._ticket_links[111] = [
            RemoteTicketLink(target_id=target.id, remote_ticket=222)
        ]
        router._close_remote_ticket = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "Cloture distante complete."}
        )

        result = asyncio.run(router.synchronize_external_close(111, profit=-3.0))
        asyncio.run(router.close())

        router._close_remote_ticket.assert_awaited_once_with(target, 222)
        self.assertTrue(result["success"])
        self.assertEqual(result["copy_summary"]["success"], 1)

    def test_synchronize_external_close_rebuilds_links_after_restart(self) -> None:
        """Verifie la reconstruction des liens avant une cloture externe positive."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 50K",
            banker_base_url="http://banker-50k:8110",
            balance_reference=Decimal("100000"),
            supported_symbols=["BTCUSD"],
        )
        router.targets[target.id] = target
        open_time = datetime(2026, 5, 10, 20, 0, 0)
        router._fetch_remote_positions = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "ticket": 222,
                    "symbol": "BTCUSD",
                    "action": "BUY",
                    "volume": 0.20,
                    "open_price": 80922.52,
                    "open_time": open_time.isoformat(),
                }
            ]
        )
        router._fetch_remote_position_snapshot = AsyncMock(  # type: ignore[method-assign]
            return_value={"ticket": 222, "volume": 0.20, "open_price": 80922.52}
        )
        router._close_remote_ticket = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "success": True,
                "ticket": 222,
                "volume_closed": 0.14,
                "volume_remaining": 0.06,
                "message": "Cloture partielle distante reussie.",
            }
        )
        router._modify_remote_ticket = AsyncMock(  # type: ignore[method-assign]
            return_value={"success": True, "ticket": 222, "message": "BE distant applique."}
        )

        result = asyncio.run(
            router.synchronize_external_close(
                111,
                profit=6.95,
                master_snapshot={
                    "symbol": "BTCUSD",
                    "action": "BUY",
                    "entry_price": 80922.52,
                    "volume": 0.02,
                    "open_time": open_time,
                },
            )
        )
        asyncio.run(router.close())

        self.assertEqual(result["copy_summary"]["success"], 1)
        self.assertEqual(result["copy_results"][0]["runner_mode"], "partial_close_break_even")
        router._close_remote_ticket.assert_awaited_once()
        router._modify_remote_ticket.assert_awaited_once_with(
            target,
            222,
            sl=80922.52,
            tp=0.0,
        )

    def test_execute_order_skips_fan_out_for_copy_source(self) -> None:
        """Verifie qu'un ordre deja copie ne repart pas en cascade."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTMO 2",
            banker_base_url="http://banker-2:8100",
        )
        router.targets[target.id] = target
        router._execute_remote_target_order = AsyncMock(
            return_value={"success": True, "ticket": 222, "message": "Ordre copie execute."}
        )

        order = TradeOrder(
            symbol="EURUSD",
            action=TradeAction.SELL,
            volume=Decimal("0.50"),
            stop_loss_price=Decimal("1.10"),
            source=OrderSource.COPY,
        )

        result = asyncio.run(router.execute_order(order))
        asyncio.run(router.close())

        self.assertTrue(result["success"])
        self.assertEqual(router._execute_remote_target_order.await_count, 0)

    def test_execute_scaled_copy_translates_symbol_with_supported_catalog(self) -> None:
        """Verifie qu'un symbole source est traduit via le catalogue cible."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTUK",
            banker_base_url="http://banker-ftuk:8120",
            allocation_ratio=Decimal("1.0"),
            supported_symbols=["GOLD", "EURUSD", "USDJPY"],
        )
        router.targets[target.id] = target
        router._resolve_target_balance = AsyncMock(return_value=Decimal("100000"))

        captured_symbols: list[str] = []

        async def fake_execute_remote_target_order(_target, order):
            captured_symbols.append(order.symbol)
            return {"success": True, "ticket": 333, "message": "Ordre copie execute."}

        router._execute_remote_target_order = AsyncMock(side_effect=fake_execute_remote_target_order)

        order = TradeOrder(
            symbol="XAUUSD",
            action=TradeAction.BUY,
            volume=Decimal("0.50"),
            stop_loss_price=Decimal("4500"),
        )

        result = asyncio.run(
            router._execute_scaled_copy(
                target=target,
                master_order=order,
                master_balance=Decimal("100000"),
            )
        )
        asyncio.run(router.close())

        self.assertTrue(result["success"])
        self.assertEqual(result["resolved_symbol"], "GOLD")
        self.assertEqual(captured_symbols, ["GOLD"])

    def test_execute_scaled_copy_refuses_unsupported_symbol(self) -> None:
        """Verifie qu'un symbole incompatible est refuse proprement."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTUK",
            banker_base_url="http://banker-ftuk:8120",
            allocation_ratio=Decimal("1.0"),
            supported_symbols=["EURUSD", "USDJPY"],
        )
        router.targets[target.id] = target
        router._resolve_target_balance = AsyncMock(return_value=Decimal("100000"))
        router._execute_remote_target_order = AsyncMock(
            return_value={"success": True, "ticket": 444, "message": "Ordre copie execute."}
        )

        order = TradeOrder(
            symbol="US100.cash",
            action=TradeAction.BUY,
            volume=Decimal("0.50"),
            stop_loss_price=Decimal("27000"),
        )

        result = asyncio.run(
            router._execute_scaled_copy(
                target=target,
                master_order=order,
                master_balance=Decimal("100000"),
            )
        )
        asyncio.run(router.close())

        self.assertFalse(result["success"])
        self.assertIn("incompatible", result["message"])
        self.assertEqual(router._execute_remote_target_order.await_count, 0)

    def test_execute_scaled_copy_resolves_broker_suffix_symbol(self) -> None:
        """Verifie qu'un suffixe broker est resolu automatiquement."""
        primary = DummyPrimaryService()
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="Compte FTUK",
            banker_base_url="http://banker-ftuk:8120",
            allocation_ratio=Decimal("1.0"),
            supported_symbols=["EURUSD.e", "XAUUSD.m", "US500.e"],
        )
        router.targets[target.id] = target
        router._resolve_target_balance = AsyncMock(return_value=Decimal("100000"))

        captured_symbols: list[str] = []

        async def fake_execute_remote_target_order(_target, order):
            captured_symbols.append(order.symbol)
            return {"success": True, "ticket": 555, "message": "Ordre copie execute."}

        router._execute_remote_target_order = AsyncMock(side_effect=fake_execute_remote_target_order)

        order = TradeOrder(
            symbol="EURUSD",
            action=TradeAction.BUY,
            volume=Decimal("0.50"),
            stop_loss_price=Decimal("1.10"),
        )

        result = asyncio.run(
            router._execute_scaled_copy(
                target=target,
                master_order=order,
                master_balance=Decimal("100000"),
            )
        )
        asyncio.run(router.close())

        self.assertTrue(result["success"])
        self.assertEqual(result["resolved_symbol"], "EURUSD.e")
        self.assertEqual(captured_symbols, ["EURUSD.e"])

    def test_repair_open_positions_creates_missing_remote_positions(self) -> None:
        """Verifie le rattrapage des followers ajoutes apres l'ouverture du panier."""
        primary = DummyPrimaryService()
        primary.open_positions = [
            Position(
                ticket=111,
                symbol="XAUUSD",
                action=TradeAction.BUY,
                volume=Decimal("0.02"),
                open_price=Decimal("4700"),
                current_price=Decimal("4710"),
                stop_loss=Decimal("4650"),
                take_profit=None,
                profit=Decimal("10"),
                open_time=datetime(2026, 5, 7, 20, 0, 0),
            )
        ]
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="FTUK 333382355",
            banker_base_url="http://banker-ftuk-355:8140",
            allocation_ratio=Decimal("1.0"),
            balance_reference=Decimal("100000"),
            symbol_map={"XAUUSD": "XAUUSD.m"},
            supported_symbols=["XAUUSD.m"],
        )
        router.targets[target.id] = target
        router._fetch_remote_positions = AsyncMock(return_value=[])
        router._resolve_target_balance = AsyncMock(return_value=Decimal("100000"))
        router._execute_remote_target_order = AsyncMock(
            return_value={"success": True, "ticket": 555, "message": "Ordre reparateur execute."}
        )

        result = asyncio.run(router.repair_open_positions())
        asyncio.run(router.close())

        self.assertTrue(result["success"])
        self.assertEqual(result["targets"][0]["created_positions"], 1)
        self.assertIn(111, router._ticket_links)
        self.assertEqual(router._ticket_links[111][0].remote_ticket, 555)

    def test_repair_open_positions_matches_existing_remote_positions(self) -> None:
        """Verifie qu'une position deja ouverte sur un follower est rattachee sans duplication."""
        primary = DummyPrimaryService()
        primary.open_positions = [
            Position(
                ticket=111,
                symbol="US100.cash",
                action=TradeAction.BUY,
                volume=Decimal("0.05"),
                open_price=Decimal("28550"),
                current_price=Decimal("28600"),
                stop_loss=Decimal("28250"),
                take_profit=None,
                profit=Decimal("15"),
                open_time=datetime(2026, 5, 7, 20, 5, 0),
            )
        ]
        router = CopyTradingRouter(primary)
        target = CopyTradingTarget(
            id=uuid4(),
            name="FTMO 50K",
            banker_base_url="http://banker-ftmo-50k:8110",
            allocation_ratio=Decimal("1.0"),
            balance_reference=Decimal("50000"),
            supported_symbols=["US100.cash"],
        )
        router.targets[target.id] = target
        router._fetch_remote_positions = AsyncMock(
            return_value=[
                {
                    "ticket": 222,
                    "symbol": "US100.cash",
                    "action": "BUY",
                    "volume": "0.24",
                    "open_price": "28549.8",
                    "open_time": "2026-05-07T20:05:02",
                }
            ]
        )
        router._resolve_target_balance = AsyncMock(return_value=Decimal("50000"))
        router._execute_remote_target_order = AsyncMock(
            return_value={"success": True, "ticket": 999, "message": "Ne doit pas etre appele."}
        )

        result = asyncio.run(router.repair_open_positions())
        asyncio.run(router.close())

        self.assertTrue(result["success"])
        self.assertEqual(result["targets"][0]["matched_positions"], 1)
        self.assertEqual(router._execute_remote_target_order.await_count, 0)
        self.assertEqual(router._ticket_links[111][0].remote_ticket, 222)


if __name__ == "__main__":
    unittest.main()
