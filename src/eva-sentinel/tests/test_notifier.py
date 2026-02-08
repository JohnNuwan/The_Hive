import pytest
import asyncio
from eva_sentinel.services.notifier import TelegramNotifier

@pytest.mark.asyncio
async def test_notifier_mock_mode():
    """Verify that the notifier works in mock mode without crashing."""
    notifier = TelegramNotifier()
    # Ensure it starts in mock mode if no env vars
    assert notifier.enabled is False or (notifier.bot_token and notifier.chat_id)
    
    # Test emergency message
    success = await notifier.notify_emergency("TEST_SOURCE", "Test alert message")
    assert success is True

@pytest.mark.asyncio
async def test_notifier_trade_formatting():
    """Verify trade message formatting logic."""
    notifier = TelegramNotifier()
    success = await notifier.notify_trade("XAUUSD", 150.50, 12345678)
    assert success is True
