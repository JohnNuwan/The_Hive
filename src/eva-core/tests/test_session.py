import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from eva_core.main import app
from uuid import UUID
from shared.internal_auth import InternalAuth

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}

@pytest.fixture
def mock_dependencies():
    # Patch all dependencies initialized in lifespan
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client") as mock_get_redis, \
         patch("eva_core.main.IntentRouter") as mock_intent_router, \
         patch("eva_core.main.get_llm_service") as mock_get_llm, \
         patch("eva_core.main.get_memory_service") as mock_get_memory, \
         patch("eva_core.main.PromptMaster") as mock_prompt_master, \
         patch("eva_core.main.EVAMQTTClient") as mock_mqtt_client_cls, \
         patch("eva_core.main.StrategyOrchestrator") as mock_strategy_orch, \
         patch("eva_core.main.SelfHealingService") as mock_self_healing, \
         patch("eva_core.main.SystemMonitor") as mock_system_monitor, \
         patch("shared.redis_client.get_redis_client") as mock_shared_get_redis:

        # Setup Redis mock
        mock_redis_instance = AsyncMock()
        mock_redis_instance.cache_get.return_value = {"status": "online", "ts": 1234567890}
        mock_redis_instance._client.keys.return_value = []

        mock_get_redis.return_value = mock_redis_instance
        mock_shared_get_redis.return_value = mock_redis_instance

        # Setup MQTT mock
        mock_mqtt_instance = AsyncMock()
        mock_mqtt_client_cls.return_value = mock_mqtt_instance

        # Setup SelfHealingService mock
        mock_self_healing_instance = MagicMock()
        # Ensure start_monitoring returns a coroutine (AsyncMock is awaitable)
        mock_self_healing_instance.start_monitoring = AsyncMock()
        mock_self_healing.return_value = mock_self_healing_instance

        yield

@pytest.fixture
def client(mock_dependencies):
    with TestClient(app) as c:
        yield c

def test_create_session(client, auth_headers):
    """
    Test creating a new session.
    Verifies that the endpoint returns a 200 OK status and a valid UUID.
    """
    response = client.post("/session", headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert "session_id" in data

    # Verify that the returned session_id is a valid UUID
    try:
        uuid_obj = UUID(data["session_id"])
        assert str(uuid_obj) == data["session_id"]
    except ValueError:
        pytest.fail("Returned session_id is not a valid UUID")
