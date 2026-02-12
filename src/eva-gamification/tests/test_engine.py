"""
Tests for Gamification Engine.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from eva_gamification.engine import GamificationEngine
from eva_gamification.models import TradeResult, GameState

@pytest.fixture
def mock_redis():
    mock = AsyncMock()
    mock.cache_get.return_value = None
    mock.cache_set.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    return mock

@pytest.fixture
def engine(mock_redis):
    with patch("eva_gamification.engine.get_redis_client", return_value=mock_redis):
        # We also need to mock ComplianceGovernor redis inside
        eng = GamificationEngine()
        # Mocking initialized redis connection in engine
        eng.redis = mock_redis
        eng.governor.redis_wrapper = mock_redis # Inject mock
        return eng

@pytest.mark.asyncio
async def test_sp_calculation(engine):
    """Verify SP formula: (Profit * 10) + (Eff * 5) - (Risk * 20)."""
    trade = TradeResult(
        ticket=123,
        profit=100.0,
        symbol="EURUSD",
        volume=0.01,
        duration_seconds=60,
        max_drawdown_percent=0.0,
        efficiency_score=1.0
    )

    # Expected SP = (100 * 10) + (1 * 5) - (0) = 1005
    sp = engine._calculate_sp(trade)
    assert sp == 1005.0

@pytest.mark.asyncio
async def test_risk_penalty(engine):
    """Verify Risk Penalty applies exponentially."""
    trade = TradeResult(
        ticket=124,
        profit=50.0,
        symbol="EURUSD",
        volume=0.01,
        duration_seconds=60,
        max_drawdown_percent=2.0, # 2% Drawdown
        efficiency_score=0.0
    )

    # Expected SP = (50 * 10) + 0 - (2^2 * 20) = 500 - 80 = 420
    sp = engine._calculate_sp(trade)
    assert sp == 420.0

@pytest.mark.asyncio
async def test_quest_progress_and_unlock(engine):
    """Verify quest progress updates and unlocking logic."""
    # Mock subprocess for Axelera detection
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "Axelera AI accelerator found"

        # Initial State
        engine.state.sp = 0
        engine.state.total_profit = 0

        # Trade that brings profit to 4000 (Target)
        trade = TradeResult(
            ticket=999,
            profit=4000.0,
            symbol="BTCUSD",
            volume=1.0,
            duration_seconds=300,
            max_drawdown_percent=0.0
        )

        result = await engine.process_trade_result(trade)

        assert engine.state.quest_progress >= 1.0
        assert "The Sight Awakening" in engine.state.unlocked_techs
        assert engine.state.level == 1
