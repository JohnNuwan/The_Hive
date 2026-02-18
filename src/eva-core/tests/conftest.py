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
    # Setup mocks for dependencies that are awaited in lifespan
    mock_mqtt_class = MagicMock()
    mock_mqtt_instance = MagicMock()
    mock_mqtt_instance.connect = AsyncMock()
    mock_mqtt_class.return_value = mock_mqtt_instance

    mock_strategy_class = MagicMock()
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.route_request = AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_self_healing_class = MagicMock()
    mock_self_healing_instance = MagicMock()
    mock_self_healing_instance.start_monitoring = AsyncMock()  # Coroutine for asyncio.create_task
    mock_self_healing_class.return_value = mock_self_healing_instance

    mock_redis_instance = MagicMock()
    mock_redis_instance.disconnect = AsyncMock()

    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis_instance), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt_instance), \
         patch("eva_core.main.StrategyOrchestrator", return_value=mock_strategy_instance), \
         patch("eva_core.main.SelfHealingService", return_value=mock_self_healing_instance), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app
        # Mock state objects (these might be overwritten by lifespan, but good for safety)
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = mock_strategy_instance
        app.state.self_healing = mock_self_healing_instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
