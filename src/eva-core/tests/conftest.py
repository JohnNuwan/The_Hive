import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before they are imported by the app
if "numpy" not in sys.modules:
    sys.modules["numpy"] = MagicMock()
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()
if "torch.nn" not in sys.modules:
    sys.modules["torch.nn"] = MagicMock()
if "torch.nn.functional" not in sys.modules:
    sys.modules["torch.nn.functional"] = MagicMock()

# Mem0
if "mem0" not in sys.modules:
    sys.modules["mem0"] = MagicMock()

# Langchain
if "langchain_ollama" not in sys.modules:
    sys.modules["langchain_ollama"] = MagicMock()

# Qdrant
if "qdrant_client" not in sys.modules:
    sys.modules["qdrant_client"] = MagicMock()
    sys.modules["qdrant_client.models"] = MagicMock()
    sys.modules["qdrant_client.http"] = MagicMock()
    sys.modules["qdrant_client.http.models"] = MagicMock()

# Neo4j
if "neo4j" not in sys.modules:
    sys.modules["neo4j"] = MagicMock()

# psutil
if "psutil" not in sys.modules:
    sys.modules["psutil"] = MagicMock()

# docker
if "docker" not in sys.modules:
    sys.modules["docker"] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from shared.internal_auth import InternalAuth

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}

@pytest.fixture
def client():
    # Prepare mocks with async methods pre-configured
    mock_redis_instance = AsyncMock()
    mock_redis_instance.disconnect = AsyncMock()
    # Defaults for test_api.py
    mock_redis_instance.cache_get.return_value = {"status": "online", "ts": 1234567890}
    mock_redis_instance.cache_mget.return_value = [{"status": "online", "ts": 1234567890}] * 7
    mock_redis_instance._client.keys.return_value = []

    mock_mqtt_instance = MagicMock()
    mock_mqtt_instance.connect = AsyncMock()

    mock_healing_instance = MagicMock()
    mock_healing_instance.start_monitoring = AsyncMock()

    mock_strategy_instance = MagicMock()
    mock_strategy_instance.route_request = AsyncMock()

    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis_instance), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt_instance), \
         patch("eva_core.main.StrategyOrchestrator", return_value=mock_strategy_instance), \
         patch("eva_core.main.SelfHealingService", return_value=mock_healing_instance), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock), \
         patch("eva_core.main.IntentRouter", new_callable=MagicMock), \
         patch("eva_core.main.PromptMaster", new_callable=MagicMock):

        from eva_core.main import app
        # Mock state objects
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        # app.state.mqtt will be set by lifespan using MockMQTT instance
        # app.state.strategy_orchestrator will be set by lifespan using MockStrategy instance
        # app.state.self_healing will be set by lifespan using MockHealing instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
