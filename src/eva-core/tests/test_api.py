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
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis), \
         patch("eva_core.main.get_memory_service", return_value=MagicMock()), \
         patch("eva_core.main.get_llm_service", return_value=MagicMock()), \
         patch("eva_core.main.EVAMQTTClient", return_value=AsyncMock()), \
         patch("eva_core.main.StrategyOrchestrator", return_value=MagicMock()), \
         patch("eva_core.main.SelfHealingService", return_value=MagicMock()), \
         patch("eva_core.main.SystemMonitor", return_value=MagicMock()), \
         patch("eva_core.main.PromptMaster", return_value=MagicMock()):

        # Ensure MQTT connect is awaited properly in lifespan
        mqtt_mock = AsyncMock()
        mqtt_mock.connect = AsyncMock()

        # Self Healing start_monitoring is a task
        healing_mock = MagicMock()
        healing_mock.start_monitoring = AsyncMock()

        with patch("eva_core.main.EVAMQTTClient", return_value=mqtt_mock), \
             patch("eva_core.main.SelfHealingService", return_value=healing_mock):

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
