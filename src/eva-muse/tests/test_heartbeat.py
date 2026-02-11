import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from eva_muse.main import hard_heartbeat

@pytest.mark.asyncio
async def test_hard_heartbeat_logs_error():
    """Verify that hard_heartbeat logs errors instead of silently passing."""

    # Mock dependencies
    with patch('eva_muse.main.get_redis_client') as mock_get_client, \
         patch('eva_muse.main.logger') as mock_logger, \
         patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

        # Setup mock redis
        # get_redis_client is synchronous in main.py, returning a RedisClient instance
        # The RedisClient instance has async methods like cache_set
        mock_redis_client = MagicMock()
        mock_get_client.return_value = mock_redis_client

        # Configure cache_set as an AsyncMock to be awaited
        mock_cache_set = AsyncMock()
        mock_redis_client.cache_set = mock_cache_set

        # Configure side effect to raise Exception first, then CancelledError to stop loop
        # The Exception should be caught and logged
        # The CancelledError should propagate and stop the test
        error_msg = "Redis connection failed"
        mock_cache_set.side_effect = [Exception(error_msg), asyncio.CancelledError()]

        # Run the function
        try:
            await hard_heartbeat()
        except asyncio.CancelledError:
            pass

        # Verify logger was called with error
        # Since we are fixing the code to log, we expect this to succeed AFTER the fix.
        # But for now (before fix), this test should fail because logger is not called.
        mock_logger.error.assert_called_with(f"Error in heartbeat: {error_msg}")
