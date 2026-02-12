
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva_banker.main import app, trigger_kill_switch
from eva_banker.services.mt5 import MT5Service


@dataclass
class Position:
    ticket: int
    symbol: str

@pytest.mark.asyncio
async def test_trigger_kill_switch_closes_all_positions():
    # Setup Mock MT5Service
    mock_mt5 = MagicMock(spec=MT5Service)

    positions = [
        Position(ticket=1, symbol="EURUSD"),
        Position(ticket=2, symbol="GBPUSD"),
        Position(ticket=3, symbol="XAUUSD")
    ]

    # Configure mock responses
    mock_mt5.get_open_positions = AsyncMock(return_value=positions)
    mock_mt5.close_position = AsyncMock(return_value={"success": True})

    # Set app state
    app.state.mt5_service = mock_mt5

    # Execute
    result = await trigger_kill_switch()

    # Verify
    assert result["status"] == "kill_switch_triggered"
    assert "3 positions fermées" in result["message"]

    # Check calls
    assert mock_mt5.get_open_positions.call_count == 1
    assert mock_mt5.close_position.call_count == 3

    # Verify call arguments
    # Note: call_args_list stores call objects, which are tuples of (args, kwargs)
    # We check args.
    actual_calls = [c.args for c in mock_mt5.close_position.call_args_list]
    # Order might not be guaranteed with asyncio.gather, so check set equality if needed
    # But for now, let's just check length and content
    assert len(actual_calls) == 3
    assert (1,) in actual_calls
    assert (2,) in actual_calls
    assert (3,) in actual_calls

@pytest.mark.asyncio
async def test_trigger_kill_switch_handles_failures():
    # Setup Mock MT5Service
    mock_mt5 = MagicMock(spec=MT5Service)

    positions = [
        Position(ticket=1, symbol="EURUSD"),
        Position(ticket=2, symbol="GBPUSD")
    ]

    mock_mt5.get_open_positions = AsyncMock(return_value=positions)
    # Fail one call
    mock_mt5.close_position = AsyncMock(side_effect=[
        {"success": True},
        {"success": False}
    ])

    app.state.mt5_service = mock_mt5

    result = await trigger_kill_switch()

    assert result["status"] == "kill_switch_triggered"
    assert "1 positions fermées sur 2" in result["message"]
    assert mock_mt5.close_position.call_count == 2

@pytest.mark.asyncio
async def test_trigger_kill_switch_handles_exceptions():
    # Setup Mock MT5Service
    mock_mt5 = MagicMock(spec=MT5Service)

    positions = [
        Position(ticket=1, symbol="EURUSD"),
        Position(ticket=2, symbol="GBPUSD")
    ]

    mock_mt5.get_open_positions = AsyncMock(return_value=positions)
    # Raise exception for one call
    mock_mt5.close_position = AsyncMock(side_effect=[
        {"success": True},
        Exception("Network Error")
    ])

    app.state.mt5_service = mock_mt5

    # We expect it to handle exceptions gracefully if we implement it right,
    # or fail if not.
    # Current implementation would crash if the exception happens.
    # New implementation with return_exceptions=True should handle it.

    # With asyncio.gather(return_exceptions=True), exceptions are returned as objects in the result list.
    # We want to ensure the kill switch continues processing other positions.
    result = await trigger_kill_switch()

    assert result["status"] == "kill_switch_triggered"
    assert "1 positions fermées" in result["message"]
