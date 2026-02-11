
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from eva_core.main import app
from shared.internal_auth import InternalAuth
from shared.config import get_settings

@pytest.fixture
def mock_redis():
    mock_client = AsyncMock()
    # Mock cache_get to return a valid status
    mock_client.cache_get.return_value = {"status": "online", "ts": 1234567890}
    # Mock keys
    mock_client._client.keys.return_value = []
    # Mock broadcast_to_swarm
    mock_client.broadcast_to_swarm = AsyncMock()
    # Mock send_to_agent
    mock_client.send_to_agent = AsyncMock()
    return mock_client

@pytest.fixture
def client(mock_redis):
    # Patch init_redis to avoid connecting to real Redis
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis):

        # Patch dependencies that are initialized in lifespan
        with patch("eva_core.main.get_memory_service") as mock_memory, \
             patch("eva_core.main.get_llm_service") as mock_llm, \
             patch("eva_core.main.IntentRouter") as mock_router, \
             patch("eva_core.main.PromptMaster") as mock_prompt_master, \
             patch("eva_core.main.EVAMQTTClient") as mock_mqtt_cls, \
             patch("eva_core.main.StrategyOrchestrator") as mock_strategy, \
             patch("eva_core.main.SelfHealingService") as mock_healing, \
             patch("eva_core.main.SystemMonitor") as mock_monitor:

            # Configure mocks
            mock_memory.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            mock_router.return_value = MagicMock()
            mock_prompt_master.return_value = MagicMock()

            mock_mqtt_instance = AsyncMock()
            mock_mqtt_instance.connect = AsyncMock()
            mock_mqtt_cls.return_value = mock_mqtt_instance

            mock_strategy.return_value = MagicMock()

            mock_healing_instance = MagicMock()
            # Mock start_monitoring to return a coroutine/task
            mock_healing_instance.start_monitoring = AsyncMock()
            mock_healing.return_value = mock_healing_instance

            mock_monitor.return_value = MagicMock()

            # Initialize app state manually (optional, but good for test isolation)
            # Lifespan will run and use the patched mocks

            with TestClient(app) as c:
                yield c

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}
