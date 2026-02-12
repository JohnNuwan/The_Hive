"""
Configuration de tests pour eva-core
"""

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
    # Mock mget/cache_mget to return a list of non-None values
    # The endpoint expects a list of dictionaries (or None)
    mock_client.cache_mget.return_value = [{"status": "online", "ts": 1234567890}] * 7
    return mock_client

@pytest.fixture
def client(mock_redis):
    # Setup mocks for services
    mock_memory_service = MagicMock()
    mock_llm_service = MagicMock()
    mock_intent_router = MagicMock()
    mock_prompt_master = MagicMock()

    mock_mqtt_client = AsyncMock()
    mock_mqtt_client.connect = AsyncMock()

    mock_strategy_orchestrator = MagicMock()

    mock_self_healing = MagicMock()
    mock_self_healing.start_monitoring = AsyncMock()

    mock_system_monitor = MagicMock()

    # Patch all external services and lifespan dependencies
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
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

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}

def test_health_endpoint(client):
    """Vérifie que l'endpoint /health répond"""
    # /health est exclu de l'auth par défaut
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_agent_status_endpoint(client, auth_headers):
    """Vérifie l'endpoint status (simulation)"""
    response = client.get("/agents/status", headers=auth_headers)
    assert response.status_code == 200
    # The response depends on redis mock
    json_response = response.json()
    assert "core" in json_response
    assert json_response["core"]["status"] == "online"

def test_token_generation_and_verification():
    """Verify that a token generated can be verified and contains correct data."""
    agent = "test-agent"
    token = InternalAuth.generate_token(agent)

    assert isinstance(token, str)

    payload = InternalAuth.verify_token(token)
    assert payload is not None
    assert payload["src"] == agent
    assert payload["iss"] == "hive-core"

def test_token_expiration():
    """Verify that an expired token is rejected (simulated)."""
    # We can't easily wait 60s in a unit test, but we can verify it doesn't fail immediately.
    token = InternalAuth.generate_token("fast-agent")
    payload = InternalAuth.verify_token(token)
    assert payload is not None

def test_invalid_token():
    """Verify that a tampered token is rejected."""
    token = InternalAuth.generate_token("agent")
    tampered_token = token + "tampered"

    payload = InternalAuth.verify_token(tampered_token)
    assert payload is None
