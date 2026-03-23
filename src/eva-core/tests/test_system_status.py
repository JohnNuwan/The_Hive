"""
Tests for system status endpoint in eva-core
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from eva_core.main import app
from shared.config import get_settings
from shared.internal_auth import InternalAuth
import shared.redis_client
import httpx

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

    mock_mqtt_client = AsyncMock()
    mock_mqtt_client.connect = AsyncMock()

    mock_strategy_orchestrator = MagicMock()

    mock_self_healing = MagicMock()
    mock_self_healing.start_monitoring = AsyncMock()

    mock_system_monitor = MagicMock()
    mock_autonomy_service = MagicMock()
    mock_autonomy_service.start_monitoring = AsyncMock()
    mock_autonomy_service.get_snapshot.return_value = {
        "generated_at": "2026-03-11T10:00:00",
        "posture": {
            "status": "ready",
            "recommended_mode": "assisted_live",
            "blockers": [],
        },
    }
    mock_autonomy_service.refresh_snapshot = AsyncMock(
        return_value=mock_autonomy_service.get_snapshot.return_value
    )

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
         patch("eva_core.main.AutonomyService", return_value=mock_autonomy_service), \
         patch("eva_core.main.SystemMonitor", return_value=mock_system_monitor):

        # Configure AsyncMocks for awaited methods
        mock_mqtt_instance = MockMQTT.return_value
        mock_mqtt_instance.connect = AsyncMock()

        mock_healing_instance = MockHealing.return_value
        mock_healing_instance.start_monitoring = AsyncMock()

        # Initialize app state manually
        app.state.settings = get_settings()
        # Override the setting for the test
        app.state.settings.sentinel_api_host = "test-sentinel-host"
        app.state.settings.sentinel_api_port = 9999

        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.prompt_master = MagicMock()
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = MagicMock()
        app.state.self_healing = mock_healing_instance
        app.state.system_monitor = MagicMock()
        app.state.autonomy_service = mock_autonomy_service

        with TestClient(app) as c:
            yield c

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}

def test_system_status_uses_configured_host(client, auth_headers):
    """
    Test that /system/status uses the configured sentinel_api_host and port.
    """
    # Mock the httpx.AsyncClient context manager used inside the endpoint
    with patch("httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client_instance

        # Setup the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cpu": 50}
        mock_client_instance.get.return_value = mock_response

        response = client.get("/system/status", headers=auth_headers)

        assert response.status_code == 200

        # Verify that httpx.AsyncClient.get was called with the correct URL
        expected_url = "http://test-sentinel-host:9999/system/metrics"
        mock_client_instance.get.assert_called_once()
        args, kwargs = mock_client_instance.get.call_args
        assert args[0] == expected_url
        assert response.json()["sentinel"]["status"] == "online"
