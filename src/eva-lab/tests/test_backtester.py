import pytest
from eva_lab.backtester import Backtester, BacktestResult

@pytest.mark.asyncio
async def test_backtest_execution():
    """Test that a backtest runs and returns a valid result."""
    backtester = Backtester()
    result = await backtester.run_backtest("TestStrategy", symbol="EURUSD", period_months=1)

    assert isinstance(result, BacktestResult)
    assert result.strategy_name == "TestStrategy"
    assert result.symbol == "EURUSD"
    assert result.total_trades >= 0
    assert len(result.equity_curve) > 0
    # The first equity point should be the initial balance
    assert result.equity_curve[0] == 10000.0

@pytest.mark.asyncio
async def test_backtest_determinism():
    """Test that two backtests with the same parameters produce identical results."""
    backtester1 = Backtester()
    result1 = await backtester1.run_backtest("TestStrategy", symbol="EURUSD", period_months=1)

    backtester2 = Backtester()
    result2 = await backtester2.run_backtest("TestStrategy", symbol="EURUSD", period_months=1)

    assert result1.total_trades == result2.total_trades
    assert result1.total_profit == result2.total_profit
    assert result1.equity_curve == result2.equity_curve
    assert result1.win_rate == result2.win_rate
    assert result1.max_drawdown == result2.max_drawdown

@pytest.mark.asyncio
async def test_history():
    """Test that backtest results are stored in history."""
    backtester = Backtester()
    await backtester.run_backtest("Strategy1", symbol="EURUSD", period_months=1)
    await backtester.run_backtest("Strategy2", symbol="GBPUSD", period_months=1)

    history = backtester.get_history()
    assert len(history) == 2
    assert history[0]["strategy"] == "Strategy1"
    assert history[1]["strategy"] == "Strategy2"
    assert history[0]["symbol"] == "EURUSD"
    assert history[1]["symbol"] == "GBPUSD"

@pytest.mark.asyncio
async def test_backtest_edge_cases():
    """Test edge cases like zero period or negative balance."""
    backtester = Backtester()

    # Zero period
    result = await backtester.run_backtest("ZeroPeriod", period_months=0)
    assert result.total_trades == 0
    assert result.win_rate == 0.0
    assert len(result.equity_curve) == 1

    # Negative initial balance
    result = await backtester.run_backtest("NegativeBalance", initial_balance=-1000.0)
    assert result.equity_curve[0] == -1000.0
    # Even with negative balance, it should run (though financially weird)
    assert result.total_trades >= 0
