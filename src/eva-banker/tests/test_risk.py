"""
Tests de gestion des risques pour The Banker
"""

import pytest
from decimal import Decimal
from eva_banker.services.risk import RiskValidator

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
    # Risque 1% sur 100k avec un SL distant de 0.0100 (100 pips en prix standard Forex)
    lot = manager.calculate_lot_size(
        balance=Decimal("100000"),
        risk_percent=Decimal("1.0"),
        sl_distance=Decimal("0.0100")
    )
    # Risk amount = 1000. Point value = 100,000. sl_distance * point_value = 0.0100 * 100,000 = 1000. Lot = 1.0
    assert lot == 1.0
    assert isinstance(lot, float)
