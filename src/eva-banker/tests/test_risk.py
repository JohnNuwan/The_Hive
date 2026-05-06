"""Tests de gestion des risques pour The Banker."""

from datetime import datetime
from decimal import Decimal

from eva_banker.services.risk import RiskValidator


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
