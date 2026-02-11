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
    # Mock broadcast_to_swarm and send_to_agent to be awaitable
    mock_client.broadcast_to_swarm = AsyncMock()
    mock_client.send_to_agent = AsyncMock()
    return mock_client

@pytest.fixture
def client(mock_redis):
    # Patch init_redis to avoid connecting to real Redis
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis), \
         patch("eva_core.main.get_memory_service", return_value=MagicMock()) as mock_mem_service, \
         patch("eva_core.main.get_llm_service", return_value=MagicMock()) as mock_llm_service, \
         patch("eva_core.main.IntentRouter", return_value=MagicMock()) as mock_router, \
         patch("eva_core.main.PromptMaster", return_value=MagicMock()) as mock_prompt, \
         patch("eva_core.main.EVAMQTTClient") as mock_mqtt_cls, \
         patch("eva_core.main.StrategyOrchestrator", return_value=MagicMock()) as mock_strategy, \
         patch("eva_core.main.SelfHealingService") as mock_healing_cls, \
         patch("eva_core.main.SystemMonitor", return_value=MagicMock()) as mock_monitor:

        # Configure mocks specifically
        # MQTT Client needs connect to be awaitable
        mock_mqtt_instance = AsyncMock()
        mock_mqtt_cls.return_value = mock_mqtt_instance

        # SelfHealingService needs start_monitoring to be awaitable
        mock_healing_instance = MagicMock()
        mock_healing_instance.start_monitoring = AsyncMock()
        mock_healing_cls.return_value = mock_healing_instance

        # Initialize app state manually to be safe, though lifespan will overwrite some
        app.state.settings = get_settings()
        app.state.intent_router = mock_router.return_value
        app.state.llm_service = mock_llm_service.return_value
        app.state.memory_service = mock_mem_service.return_value
        app.state.prompt_master = mock_prompt.return_value
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = mock_strategy.return_value
        app.state.self_healing = mock_healing_instance
        app.state.system_monitor = mock_monitor.return_value

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
    assert "core" in response.json()
