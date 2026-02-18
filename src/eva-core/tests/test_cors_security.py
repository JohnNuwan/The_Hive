import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from eva_core.main import app
from shared.config import get_settings
import shared.redis_client

@pytest.fixture
def mock_redis():
    mock_client = AsyncMock()
    mock_client.cache_get.return_value = {"status": "online", "ts": 1234567890}
    mock_client._client.keys.return_value = []
    mock_client.cache_mget.return_value = [{"status": "online", "ts": 1234567890}] * 7
    return mock_client

@pytest.fixture
def client(mock_redis):
    # Setup mocks for services
    mock_memory_service = MagicMock()
    mock_llm_service = MagicMock()
    mock_intent_router = MagicMock()
    mock_prompt_master = MagicMock()
    mock_strategy_orchestrator = MagicMock()
    mock_system_monitor = MagicMock()

    # Patch all external services and lifespan dependencies
    with patch.object(shared.redis_client, "_redis_client", mock_redis), \
         patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis), \
         patch("eva_core.memory_layer.Memory"), \
         patch("eva_core.services.memory.MemoryService"), \
         patch("mem0.Memory"), \
         patch("eva_core.main.get_memory_service", return_value=mock_memory_service), \
         patch("eva_core.main.get_llm_service", return_value=mock_llm_service), \
         patch("eva_core.main.IntentRouter", return_value=mock_intent_router), \
         patch("eva_core.main.PromptMaster", return_value=mock_prompt_master), \
         patch("eva_core.main.EVAMQTTClient") as MockMQTT, \
         patch("eva_core.main.StrategyOrchestrator", return_value=mock_strategy_orchestrator), \
         patch("eva_core.main.SelfHealingService") as MockHealing, \
         patch("eva_core.main.SystemMonitor", return_value=mock_system_monitor):

        # Configure AsyncMocks for awaited methods
        mock_mqtt_instance = MockMQTT.return_value
        mock_mqtt_instance.connect = AsyncMock()

        mock_healing_instance = MockHealing.return_value
        mock_healing_instance.start_monitoring = AsyncMock()

        # Initialize app state manually to avoid relying on lifespan if it fails or if test client behaves oddly
        # Even though lifespan runs, our patches ensure it uses mocks
        app.state.settings = get_settings()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.prompt_master = MagicMock()
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = MagicMock()
        app.state.self_healing = mock_healing_instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c

def test_cors_allowed_origin(client: TestClient):
    """
    Test that a request from an allowed origin receives the correct CORS headers.
    Default allowed origins are ["http://localhost:3001", "http://localhost:8080"].
    """
    origin = "http://localhost:3001"
    response = client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"

def test_cors_disallowed_origin(client: TestClient):
    """
    Test that a request from a disallowed origin does not receive CORS headers allowing it.
    """
    origin = "http://evil.com"
    response = client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200
    # If the origin is disallowed, Access-Control-Allow-Origin should NOT appear
    # OR it should not match the requested origin.
    if "access-control-allow-origin" in response.headers:
        assert response.headers["access-control-allow-origin"] != origin
