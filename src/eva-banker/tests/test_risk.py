"""
Tests de gestion des risques pour The Banker
"""

import pytest
from eva_banker.services.risk import RiskManager

def test_drawdown_limit():
    """Vérifie que le RiskManager détecte un drawdown excessif"""
    manager = RiskManager(max_daily_drawdown_percent=4.0)
    
    # Cas OK : 2% de perte
    assert manager.is_risk_acceptable(current_equity=9800, balance_start_day=10000) is True
    
    # Cas Critique : 5% de perte
    assert manager.is_risk_acceptable(current_equity=9400, balance_start_day=10000) is False

def test_lot_size_calculation():
    """Vérifie le calcul de taille de lot (exemple simplifié)"""
    manager = RiskManager()
    # Risque 1% sur 100k avec stop loss de 100 pips
    # Formule simplifiée pour le test
    lot = manager.calculate_lot_size(balance=100000, risk_percent=1.0, stop_loss_pips=100)
    assert lot > 0
    assert isinstance(lot, float)
