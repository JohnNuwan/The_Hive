"""
Configuration de tests pour eva-core
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from eva_core.main import app
from shared.config import get_settings
from shared.internal_auth import InternalAuth


@pytest.fixture
def mock_redis():
    mock_client = AsyncMock()
    # Mock cache_get to return a valid status
    mock_client.cache_get.return_value = {"status": "online", "ts": 1234567890}
    # Mock cache_mget to return a list of valid statuses
    mock_client.cache_mget.return_value = [{"status": "online", "ts": 1234567890} for _ in range(7)]
    # Mock keys
    mock_client._client.keys.return_value = []
    return mock_client

@pytest.fixture
def client(mock_redis):
    # Prepare mocks for services
    mock_self_healing = MagicMock()
    mock_self_healing.start_monitoring = AsyncMock()

    mock_mqtt = AsyncMock()
    mock_mqtt.connect = AsyncMock()

    # Patch init_redis to avoid connecting to real Redis
    # Also patch services instantiated in lifespan
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis), \
         patch("eva_core.main.get_memory_service", return_value=MagicMock()), \
         patch("eva_core.main.get_llm_service", return_value=MagicMock()), \
         patch("eva_core.main.IntentRouter", return_value=MagicMock()), \
         patch("eva_core.main.PromptMaster", return_value=MagicMock()), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt), \
         patch("eva_core.main.StrategyOrchestrator", return_value=MagicMock()), \
         patch("eva_core.main.SelfHealingService", return_value=mock_self_healing), \
         patch("eva_core.main.SystemMonitor", return_value=MagicMock()):

        # Initialize app state manually (though lifespan will overwrite some, it's safer)
        app.state.settings = get_settings()

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
