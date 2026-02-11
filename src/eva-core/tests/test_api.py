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
    # Prepare mocks for lifespan services
    mock_memory_service = MagicMock()
    mock_llm_service = MagicMock()
    mock_intent_router = MagicMock()
    mock_prompt_master = MagicMock()

    mock_mqtt_client = MagicMock()
    mock_mqtt_client.connect = AsyncMock() # connect is awaited

    mock_strategy_orchestrator = MagicMock()

    mock_self_healing = MagicMock()
    mock_self_healing.start_monitoring = AsyncMock() # scheduled as task

    mock_system_monitor = MagicMock()

    # Patch everything used in lifespan
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis), \
         patch("eva_core.main.get_memory_service", return_value=mock_memory_service), \
         patch("eva_core.main.get_llm_service", return_value=mock_llm_service), \
         patch("eva_core.main.IntentRouter", return_value=mock_intent_router), \
         patch("eva_core.main.PromptMaster", return_value=mock_prompt_master), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt_client), \
         patch("eva_core.main.StrategyOrchestrator", return_value=mock_strategy_orchestrator), \
         patch("eva_core.main.SelfHealingService", return_value=mock_self_healing), \
         patch("eva_core.main.SystemMonitor", return_value=mock_system_monitor):

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
