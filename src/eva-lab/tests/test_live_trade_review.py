"""Tests du scoring des trades live fermes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eva_lab.live_trade_review import LiveTradeReviewService


class LiveTradeReviewServiceTests(unittest.TestCase):
    """Verifie la classification et les hints runtime des trades live."""

    def setUp(self) -> None:
        """Cree un service de revue isole pour chaque test."""

        self._tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tempdir.name)
        self.service = LiveTradeReviewService(
            data_dir=str(self.data_dir),
            rolling_days=5,
            min_trades_for_hints=2,
        )

    def tearDown(self) -> None:
        """Nettoie le repertoire temporaire associe au test."""

        self._tempdir.cleanup()

    def test_record_closed_trade_classifies_range_entry_loss(self) -> None:
        """Classe une perte en range comme hard negative dedie."""

        result = self.service.record_closed_trade(
            symbol="EURUSD",
            action="BUY",
            price=1.1000,
            volume=0.04,
            pnl=-12.5,
            metadata={
                "entry_price": 1.1050,
                "exit_price": 1.1000,
                "context_label": "RANGING",
                "range_context": True,
                "close_reason": "managed_loss_close",
            },
            timestamp="2026-04-30T10:00:00",
        )

        self.assertEqual(result["quality_label"], "range_entry_loss")
        self.assertEqual(result["trade_bucket"], "hard_negative")

    def test_record_closed_trade_classifies_hard_stop_exit(self) -> None:
        """Priorise la sortie sur stop dans les etiquettes negatives."""

        result = self.service.record_closed_trade(
            symbol="US100.cash",
            action="BUY",
            price=27050.0,
            volume=0.01,
            pnl=-8.0,
            metadata={
                "entry_price": 27120.0,
                "exit_price": 27050.0,
                "close_reason": "stop_loss",
                "hard_stop_exit": True,
            },
            timestamp="2026-04-30T11:00:00",
        )

        self.assertEqual(result["quality_label"], "hard_stop_exit")
        self.assertEqual(result["trade_bucket"], "hard_negative")

    def test_summary_builds_runtime_hints_from_trade_window(self) -> None:
        """Construit des hints promoteur/demoteur par symbole."""

        self.service.record_closed_trade(
            symbol="XAUUSD",
            action="BUY",
            price=4560.0,
            volume=0.01,
            pnl=18.0,
            metadata={
                "entry_price": 4540.0,
                "exit_price": 4560.0,
                "close_reason": "managed_profit_close",
            },
            timestamp="2026-04-30T12:00:00",
        )
        self.service.record_closed_trade(
            symbol="XAUUSD",
            action="BUY",
            price=4572.0,
            volume=0.01,
            pnl=9.0,
            metadata={
                "entry_price": 4560.0,
                "exit_price": 4572.0,
                "close_reason": "managed_profit_close",
            },
            timestamp="2026-04-30T13:00:00",
        )
        self.service.record_closed_trade(
            symbol="EURUSD",
            action="SELL",
            price=1.1680,
            volume=0.04,
            pnl=-6.0,
            metadata={
                "entry_price": 1.1660,
                "exit_price": 1.1680,
                "context_label": "RANGING",
                "range_context": True,
                "close_reason": "managed_loss_close",
            },
            timestamp="2026-04-30T14:00:00",
        )
        self.service.record_closed_trade(
            symbol="EURUSD",
            action="SELL",
            price=1.1690,
            volume=0.04,
            pnl=-4.5,
            metadata={
                "entry_price": 1.1670,
                "exit_price": 1.1690,
                "close_reason": "managed_loss_close",
            },
            timestamp="2026-04-30T15:00:00",
        )
        self.service.record_closed_trade(
            symbol="XAUUSD",
            action="BUY",
            price=4584.0,
            volume=0.01,
            pnl=6.0,
            metadata={
                "entry_price": 4574.0,
                "exit_price": 4584.0,
                "close_reason": "managed_profit_close",
            },
            timestamp="2026-04-30T16:00:00",
        )

        summary = self.service.get_summary()
        hints = summary["ga_runtime_hints"]

        self.assertTrue(hints["eligible"])
        self.assertIn("XAUUSD", hints["promote_symbols"])
        self.assertIn("EURUSD", hints["demote_symbols"])
        self.assertGreater(summary["hard_negative_mix"]["counts"]["range_entry_loss"], 0)


if __name__ == "__main__":
    unittest.main()
