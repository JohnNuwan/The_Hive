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


def test_eva_close_comment_is_treated_as_hold_even_before_sl_refresh():
    """Verifie qu'un reliquat EVA Close n'est pas recompte comme une entree active."""
    validator = RiskValidator(max_open_positions=3)
    positions = [
        Position(
            ticket=1,
            symbol="US30.cash",
            action=TradeAction.BUY,
            volume=Decimal("0.03"),
            open_price=Decimal("49800"),
            current_price=Decimal("49950"),
            stop_loss=None,
            profit=Decimal("45"),
            comment="EVA Close",
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


def test_gains_protection_profit_target_lock():
    """Verifie que le RiskValidator bloque et ferme tout une fois la cible de profit atteinte."""
    validator = RiskValidator(
        max_daily_profit=Decimal("2.0"),
        giveback_activation=Decimal("1.5"),
        giveback_tolerance=Decimal("0.5"),
    )
    validator.update_account_balance(Decimal("10000"))

    # Cas 1 : Gains totaux (ferme + latent) sous la cible (1.0% de profit)
    status = validator.check_gains_protection(Decimal("100"))
    assert status["lock_triggered"] is False
    assert validator._daily_profit_locked is False

    # Cas 2 : Les gains atteignent ou depassent la cible (2.5% de profit)
    status = validator.check_gains_protection(Decimal("250"))
    assert status["lock_triggered"] is True
    assert status["reason"] == "daily_profit_target_reached"
    assert validator._daily_profit_locked is True

    # Cas 3 : Les nouvelles verifications restent verrouillees
    status2 = validator.check_gains_protection(Decimal("50"))
    assert status2["lock_triggered"] is True


def test_gains_protection_giveback_trailing_protector():
    """Verifie le declenchement de la protection anti-giveback apres franchissement du pic."""
    validator = RiskValidator(
        max_daily_profit=Decimal("5.0"),
        giveback_activation=Decimal("1.5"),
        giveback_tolerance=Decimal("0.5"),
    )
    validator.update_account_balance(Decimal("10000"))

    # Cas 1 : Gains montent a 1.2% (sous le seuil d'activation de 1.5%)
    status1 = validator.check_gains_protection(Decimal("120"))
    assert status1["lock_triggered"] is False
    assert validator._daily_giveback_locked is False
    assert validator._daily_pnl_peak == Decimal("120")

    # Cas 2 : Les gains montent a 1.8% (active l'anti-giveback et etablit le pic)
    status2 = validator.check_gains_protection(Decimal("180"))
    assert status2["lock_triggered"] is False
    assert validator._daily_pnl_peak == Decimal("180")

    # Cas 3 : Les gains retombent legement a 1.5% (drop de 0.3%, inferieur a la tolerance de 0.5%)
    status3 = validator.check_gains_protection(Decimal("150"))
    assert status3["lock_triggered"] is False
    assert validator._daily_giveback_locked is False

    # Cas 4 : Les gains retombent a 1.1% (drop de 0.7%, superieur a la tolerance de 0.5%)
    status4 = validator.check_gains_protection(Decimal("110"))
    assert status4["lock_triggered"] is True
    assert status4["reason"] == "daily_giveback_triggered"
    assert validator._daily_giveback_locked is True


def test_floating_pnl_drawdown_limit():
    """Verifie que le RiskValidator detecte le drawdown induit par le P&L latent (flottant)."""
    manager = RiskValidator(max_daily_drawdown=Decimal("4.0"))
    manager.update_account_balance(Decimal("10000"))

    # Cas 1 : Aucun drawdown latent, aucun drawdown ferme
    positions_ok = [
        Position(
            ticket=1,
            symbol="EURUSD",
            action=TradeAction.BUY,
            volume=Decimal("0.10"),
            open_price=Decimal("1.1000"),
            current_price=Decimal("1.1050"),
            profit=Decimal("50"),  # Gain de $50
            open_time=datetime.now(),
        )
    ]
    manager.update_positions_snapshot(positions_ok)
    assert manager._get_daily_drawdown_percent() == Decimal("0")

    # Cas 2 : Perte latente de $200 (2% de la balance), pas de drawdown fermé
    positions_latent_drawdown = [
        Position(
            ticket=1,
            symbol="EURUSD",
            action=TradeAction.BUY,
            volume=Decimal("0.10"),
            open_price=Decimal("1.1000"),
            current_price=Decimal("1.0800"),
            profit=Decimal("-200"),  # Perte latente de $200
            open_time=datetime.now(),
        )
    ]
    manager.update_positions_snapshot(positions_latent_drawdown)
    assert manager._get_daily_drawdown_percent() == Decimal("2.00")

    # Cas 3 : Perte latente de $300 + perte fermée existante de $150 (Total $450 = 4.5% de drawdown)
    manager.record_trade_result(Decimal("-150"))  # Perte fermée de $150
    positions_critical = [
        Position(
            ticket=2,
            symbol="GBPUSD",
            action=TradeAction.SELL,
            volume=Decimal("0.15"),
            open_price=Decimal("1.3000"),
            current_price=Decimal("1.3200"),
            profit=Decimal("-300"),  # Perte latente de $300
            open_time=datetime.now(),
        )
    ]
    manager.update_positions_snapshot(positions_critical)
    assert manager._get_daily_drawdown_percent() == Decimal("4.50")


