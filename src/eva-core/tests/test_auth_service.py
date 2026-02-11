import json
from unittest.mock import AsyncMock

import pytest

from eva_core.services.auth import AuthService


@pytest.fixture
def mock_redis():
    mock_client = AsyncMock()
    # Mock underlying redis client
    mock_client._client = AsyncMock()
    return mock_client


@pytest.mark.asyncio
async def test_list_users_uses_mget(mock_redis):
    # Setup data
    user_data = {
        "username": "testuser",
        "role": "viewer",
        "display_name": "Test User",
        "created_at": "2023-01-01T00:00:00",
        "is_active": True,
    }
    keys = ["hive:auth:user:testuser"]

    # Mock keys return
    mock_redis._client.keys.return_value = keys
    # Mock mget return
    mock_redis._client.mget.return_value = [json.dumps(user_data)]

    # Initialize service
    service = AuthService(mock_redis, "secret")

    # Call method
    users = await service.list_users()

    # Assertions
    assert len(users) == 1
    assert users[0].username == "testuser"

    # Verify mget was called instead of get in a loop
    mock_redis._client.mget.assert_called_once_with(keys)
    mock_redis._client.get.assert_not_called()


@pytest.mark.asyncio
async def test_list_users_empty(mock_redis):
    # Mock keys return empty
    mock_redis._client.keys.return_value = []

    # Initialize service
    service = AuthService(mock_redis, "secret")

    # Call method
    users = await service.list_users()

    # Assertions
    assert len(users) == 0

    # Verify mget was NOT called
    mock_redis._client.mget.assert_not_called()
