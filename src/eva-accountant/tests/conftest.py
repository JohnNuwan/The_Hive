import sys
import pytest
from unittest.mock import MagicMock, AsyncMock

# Mock heavy dependencies BEFORE any import
# This is crucial for CI environments where these libs might not be present
for module in ["numpy", "torch", "pandas", "scipy", "langchain", "neo4j"]:
    sys.modules[module] = MagicMock()

from shared.internal_auth import InternalAuth

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.cache_get.return_value = {"status": "online", "ts": 1234567890}
    mock_client.cache_set = AsyncMock()
    # Support mget returning a list of None or dicts
    mock_client.cache_mget = AsyncMock(return_value=[])
    return mock_client

@pytest.fixture
def auth_headers():
    """Generate valid internal authentication headers."""
    # We use a test agent name
    token = InternalAuth.generate_token("test-accountant")
    return {"X-Hive-Internal-Token": token}
