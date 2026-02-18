"""
Tests de gestion des risques pour The Banker
"""

import pytest
from decimal import Decimal
from unittest.mock import patch
from eva_banker.services.risk import RiskValidator
from shared import TradeOrder, TradeAction, OrderType, OrderSource

# Fixture pour créer un ordre de base
@pytest.fixture
def base_order():
    return TradeOrder(
        symbol="XAUUSD",
        action=TradeAction.BUY,
        volume=Decimal("1.0"),
        stop_loss_price=Decimal("2070.0"),
        order_type=OrderType.MARKET,
        source=OrderSource.API
    )

def test_drawdown_limit():
    """Vérifie que le RiskValidator détecte un drawdown excessif"""
    manager = RiskValidator(max_daily_drawdown=Decimal("4.0"))
    manager.update_account_balance(Decimal("10000"))
    
    # Cas OK : 2% de perte
    manager.record_trade_result(Decimal("-200"))
    # Daily PnL = -200, Balance = 9800
    # DD% = 200 / 10000 = 2%
    assert manager._get_daily_drawdown_percent() < Decimal("4.0")
    
    # Cas Critique : 5% de perte (total)
    manager.record_trade_result(Decimal("-300"))
    # Daily PnL = -500
    # DD% = 500 / 10000 = 5%
    assert manager._get_daily_drawdown_percent() >= Decimal("4.0")

def test_lot_size_calculation():
    """Vérifie le calcul de taille de lot (exemple simplifié)"""
    manager = RiskValidator()
    # Risque 1% sur 100k avec stop loss de 100 pips
    lot = manager.calculate_lot_size(
        balance=Decimal("100000"),
        risk_percent=Decimal("1.0"),
        stop_loss_pips=Decimal("100")
    )
    # Risk amount = 1000. SL points value = 100 * 10 = 1000. Lot = 1.0
    assert lot == 1.0
    assert isinstance(lot, float)

@pytest.mark.asyncio
async def test_validate_order_missing_stop_loss(base_order):
    """Vérifie que l'absence de Stop Loss rejette l'ordre"""
    manager = RiskValidator()
    base_order.stop_loss_price = None

    result = await manager.validate_order(base_order)

    assert result["allowed"] is False
    assert result["reason"] == "SL obligatoire (ROE Trading)"
    assert ("stop_loss", False, "Stop Loss manquant") in result["checks"]

@pytest.mark.asyncio
async def test_validate_order_risk_too_high(base_order):
    """Vérifie le rejet si le risque par trade est trop élevé"""
    manager = RiskValidator(max_risk_per_trade=Decimal("1.0"))
    manager.update_account_balance(Decimal("10000"))

    # Configuration pour dépasser 1% de risque
    # Prix actuel (mocké dans le service) = 2080.00
    # SL = 2060.00 => Distance = 20
    # Volume = 1.0, Pip Value = 10
    # Perte potentielle = 20 * 1.0 * 10 = 200
    # Risque % = (200 / 10000) * 100 = 2.0% > 1.0%

    base_order.stop_loss_price = Decimal("2060.0")
    base_order.volume = Decimal("1.0")

    result = await manager.validate_order(base_order)

    assert result["allowed"] is False
    assert "Risque" in result["reason"]
    assert result["risk_percent"] > Decimal("1.0")

@pytest.mark.asyncio
async def test_validate_order_anti_tilt_active(base_order):
    """Vérifie que l'Anti-Tilt actif bloque les ordres"""
    manager = RiskValidator(anti_tilt_hours=1)
    manager._activate_anti_tilt()

    result = await manager.validate_order(base_order)

    assert result["allowed"] is False
    assert "Anti-Tilt actif" in result["reason"]

@pytest.mark.asyncio
async def test_validate_order_daily_drawdown_limit(base_order):
    """Vérifie la limite de drawdown journalier"""
    manager = RiskValidator(max_daily_drawdown=Decimal("4.0"))
    manager.update_account_balance(Decimal("10000"))

    # Simuler une perte > 4% (400)
    manager.record_trade_result(Decimal("-500"))

    result = await manager.validate_order(base_order)

    assert result["allowed"] is False
    assert "Drawdown journalier limite atteinte" in result["reason"]

@pytest.mark.asyncio
async def test_validate_order_total_drawdown_limit(base_order):
    """Vérifie la limite de drawdown total"""
    manager = RiskValidator(max_total_drawdown=Decimal("8.0"), max_daily_drawdown=Decimal("20.0"))
    manager.update_account_balance(Decimal("10000"))

    # Simuler une perte > 8% (900)
    manager.record_trade_result(Decimal("-900"))

    result = await manager.validate_order(base_order)

    assert result["allowed"] is False
    assert "Drawdown total limite atteinte" in result["reason"]

@pytest.mark.asyncio
async def test_validate_order_max_positions_limit(base_order):
    """Vérifie la limite de positions ouvertes"""
    manager = RiskValidator(max_open_positions=3)
    manager.update_positions_count(3)

    result = await manager.validate_order(base_order)

    assert result["allowed"] is False
    assert "Max positions atteint" in result["reason"]

@pytest.mark.asyncio
@patch("eva_banker.services.risk.calculate_var")
async def test_validate_order_var_check_failure(mock_calculate_var, base_order):
    """Vérifie le rejet si la VaR est trop élevée (marché instable)"""
    manager = RiskValidator()
    manager.update_account_balance(Decimal("100000"))

    # Risque % OK
    base_order.stop_loss_price = Decimal("2079.0")

    # Mock VaR pour déclencher l'alerte (< -0.025)
    mock_calculate_var.return_value = -0.05

    result = await manager.validate_order(base_order)

    assert result["allowed"] is False
    assert "VaR trop élevée" in result["reason"]

@pytest.mark.asyncio
@patch("eva_banker.services.risk.calculate_var")
async def test_validate_order_success(mock_calculate_var, base_order):
    """Vérifie le succès (Happy Path)"""
    manager = RiskValidator(max_risk_per_trade=Decimal("5.0"))
    manager.update_account_balance(Decimal("100000"))

    # Risque OK : Perte ~100 sur 100k (0.1%)
    base_order.stop_loss_price = Decimal("2070.0")

    # Mock VaR OK
    mock_calculate_var.return_value = -0.01

    result = await manager.validate_order(base_order)

    assert result["allowed"] is True
    assert result["reason"] is None
