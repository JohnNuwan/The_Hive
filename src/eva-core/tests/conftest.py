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

@pytest.fixture
def auth_headers():
    """Returns headers for internal authentication."""
    from shared.internal_auth import InternalAuth
    token = InternalAuth.generate_token("test-service")
    return {"X-Hive-Internal-Token": token}

@pytest.fixture
def client():
    # Setup mocks for class instances
    mqtt_mock_instance = MagicMock()
    mqtt_mock_instance.connect = AsyncMock() # connect() must be awaitable

    strategy_mock_instance = MagicMock()
    # StrategyOrchestrator methods called in endpoints might need to be async if awaited
    strategy_mock_instance.route_request = AsyncMock()

    self_healing_mock_instance = MagicMock()
    # start_monitoring() returns a coroutine for create_task
    self_healing_mock_instance.start_monitoring = AsyncMock()
    # Ensure circuit_breaker attribute exists and is accessible
    self_healing_mock_instance.circuit_breaker = None

    # Setup redis_client mock
    redis_client_mock = MagicMock()
    redis_client_mock.disconnect = AsyncMock()
    # Some other async methods might be called on redis_client too, like cache_get/set
    redis_client_mock.cache_get = AsyncMock()
    redis_client_mock.cache_mget = AsyncMock() # Used in /agents/status
    redis_client_mock.cache_set = AsyncMock()
    redis_client_mock.publish = AsyncMock()
    # get_redis_client returns this instance
    get_redis_client_mock = MagicMock(return_value=redis_client_mock)

    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", get_redis_client_mock), \
         patch("eva_core.main.EVAMQTTClient", return_value=mqtt_mock_instance), \
         patch("eva_core.main.StrategyOrchestrator", return_value=strategy_mock_instance), \
         patch("eva_core.main.SelfHealingService", return_value=self_healing_mock_instance), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app
        # Mock state objects (in case they are accessed directly, though lifespan sets them)
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        # Ensure state objects match what lifespan will set (mostly for consistency)
        app.state.mqtt = mqtt_mock_instance
        app.state.strategy_orchestrator = strategy_mock_instance
        app.state.self_healing = self_healing_mock_instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
