"""Tests `unittest` du routeur de copy trading multi-instances."""

import asyncio
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

from eva_banker.services.copy_trading import CopyTradingRouter, CopyTradingTarget
from shared import AccountBalance, OrderSource, TradeAction, TradeOrder


class DummyPrimaryService:
    """Double simple du service MT5 principal pour les tests."""

    def __init__(self) -> None:
        self.is_connected = True
        self.mock_mode = True
        self.executed_orders: list[TradeOrder] = []
        self.closed_tickets: list[int] = []
        self.modified_tickets: list[tuple[int, float, float]] = []

    async def execute_order(self, order: TradeOrder) -> dict[str, object]:
        """Memorise l'ordre local et retourne un ticket stable."""
        self.executed_orders.append(order)
        return {
            "success": True,
            "ticket": 111,
            "message": "Ordre local execute.",
        }

    async def close_position(self, ticket: int) -> dict[str, object]:
        """Memorise la cloture locale."""
        self.closed_tickets.append(ticket)
        return {"success": True, "ticket": ticket, "message": "Position locale fermee."}

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


if __name__ == "__main__":
    unittest.main()
