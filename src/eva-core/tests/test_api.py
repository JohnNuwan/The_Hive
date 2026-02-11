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
    return mock_client

@pytest.fixture
def client(mock_redis):
    # Patch init_redis to avoid connecting to real Redis
    # Also patch all lifespan services to avoid real initialization (especially Mem0/OpenAI)
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis), \
         patch("eva_core.main.get_memory_service") as mock_get_memory, \
         patch("eva_core.main.get_llm_service") as mock_get_llm, \
         patch("eva_core.main.IntentRouter") as MockIntentRouter, \
         patch("eva_core.main.PromptMaster") as MockPromptMaster, \
         patch("eva_core.main.EVAMQTTClient") as MockEVAMQTTClient, \
         patch("eva_core.main.StrategyOrchestrator") as MockStrategyOrchestrator, \
         patch("eva_core.main.SelfHealingService") as MockSelfHealingService, \
         patch("eva_core.main.SystemMonitor") as MockSystemMonitor:

        # Configure mocks
        mock_get_memory.return_value = MagicMock()
        mock_get_llm.return_value = MagicMock()

        # MQTT Client needs connect to be awaitable
        mock_mqtt_instance = AsyncMock()
        MockEVAMQTTClient.return_value = mock_mqtt_instance

        # SelfHealingService needs start_monitoring to be awaitable (created as task)
        mock_self_healing_instance = MagicMock()
        mock_self_healing_instance.start_monitoring = AsyncMock()
        MockSelfHealingService.return_value = mock_self_healing_instance

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
