"""Tests de gestion des risques pour The Banker."""

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from eva_banker.services.risk import RiskValidator, resolve_effective_max_open_positions
from shared import Position, TradeAction


def test_drawdown_limit():
    """Verifie que le RiskValidator detecte un drawdown excessif."""
    manager = RiskValidator(max_daily_drawdown=Decimal("4.0"))
    manager.update_account_balance(Decimal("10000"))

    manager.record_trade_result(Decimal("-200"))
    assert manager._get_daily_drawdown_percent() < Decimal("4.0")

    manager.record_trade_result(Decimal("-300"))
    assert manager._get_daily_drawdown_percent() >= Decimal("4.0")


def test_lot_size_calculation():
    """Verifie le calcul de taille de lot sur une paire Forex."""
    manager = RiskValidator()
    lot = manager.calculate_lot_size(
        balance=Decimal("100000"),
        risk_percent=Decimal("1.0"),
        sl_distance=Decimal("0.0100"),
        symbol="EURUSD",
    )
    assert lot == 1.0
    assert isinstance(lot, float)


def test_lot_size_uses_mt5_tick_value_for_cfd():
    """Verifie que le sizing CFD suit les economics MT5 du symbole."""
    manager = RiskValidator()
    lot = manager.calculate_lot_size(
        balance=Decimal("10000"),
        risk_percent=Decimal("1.0"),
        sl_distance=Decimal("100"),
        symbol="US100.cash",
        sizing_hint={
            "tick_size": Decimal("1"),
            "tick_value": Decimal("1"),
            "volume_min": Decimal("0.01"),
            "volume_step": Decimal("0.01"),
            "volume_max": Decimal("10.0"),
        },
    )

    assert lot == 1.0


def test_weekend_session_blocks_fx_but_not_crypto(monkeypatch):
    """Verifie que le week-end bloque le Forex mais pas la crypto."""
    manager = RiskValidator()
    manager.register_symbol_universe({"EURUSD": "forex", "BTCUSD": "crypto"})
    monkeypatch.setattr(manager, "_get_market_now", lambda: datetime(2026, 3, 7, 12, 0, 0))

    assert manager.is_within_trading_session("EURUSD") is False
    assert manager.is_within_trading_session("BTCUSD") is True


def test_follower_uses_dedicated_open_positions_limit():
    """Verifie qu'un follower herite d'un plafond de positions plus large."""
    settings = SimpleNamespace(
        banker_follower_mode=True,
        risk_max_open_positions=5,
        risk_follower_max_open_positions=12,
    )

    assert resolve_effective_max_open_positions(settings) == 12


def test_break_even_runner_is_excluded_from_open_positions_limit():
    """Verifie qu'un runner protege au break-even devient un HOLD."""
    validator = RiskValidator(max_open_positions=3)
    positions = [
        Position(
            ticket=1,
            symbol="XAUUSD",
            action=TradeAction.BUY,
            volume=Decimal("0.02"),
            open_price=Decimal("4700"),
            current_price=Decimal("4715"),
            stop_loss=Decimal("4700"),
            profit=Decimal("15"),
            comment="MZ HOLD RUNNER",
            open_time=datetime.now(),
        ),
        Position(
            ticket=2,
            symbol="EURUSD",
            action=TradeAction.SELL,
            volume=Decimal("0.10"),
            open_price=Decimal("1.1700"),
            current_price=Decimal("1.1680"),
            stop_loss=Decimal("1.1750"),
            profit=Decimal("20"),
            open_time=datetime.now(),
        ),
    ]

    validator.update_positions_snapshot(positions)

    assert validator.get_counted_open_positions() == 1
    assert validator.get_total_open_positions() == 2
    assert validator.get_hold_positions_count() == 1


def test_uncommented_manual_position_can_be_ignored_for_follower_limit():
    """Verifie qu'un follower peut ignorer les positions manuelles sans commentaire."""
    validator = RiskValidator(max_open_positions=3)
    validator.settings = SimpleNamespace(risk_ignore_uncommented_positions=True)
    positions = [
        Position(
            ticket=1,
            symbol="USOIL.cash",
            action=TradeAction.BUY,
            volume=Decimal("0.20"),
            open_price=Decimal("96"),
            current_price=Decimal("95"),
            stop_loss=None,
            profit=Decimal("-10"),
            comment="",
            open_time=datetime.now(),
        ),
        Position(
            ticket=2,
            symbol="BTCUSD",
            action=TradeAction.BUY,
            volume=Decimal("0.09"),
            open_price=Decimal("80000"),
            current_price=Decimal("80100"),
            stop_loss=Decimal("79000"),
            profit=Decimal("15"),
            comment="COPY MZ-SCP-CH-v",
            open_time=datetime.now(),
        ),
    ]

    validator.update_positions_snapshot(positions)

    assert validator.get_counted_open_positions() == 1
    assert validator.get_total_open_positions() == 2
    assert validator.get_ignored_positions_count() == 1
