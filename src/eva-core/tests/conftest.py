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
    # Patch dependencies in lifespan or global scope
    # Use standard patches (default new_callable is MagicMock, but we need to configure return_value)
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client") as MockRedisClient, \
         patch("eva_core.main.EVAMQTTClient") as MockMQTT, \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService") as MockHealing, \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app

        # Configure instance mocks to be awaitable where needed
        # EVAMQTTClient
        mock_mqtt_instance = MockMQTT.return_value
        mock_mqtt_instance.connect = AsyncMock()
        mock_mqtt_instance.publish = AsyncMock()

        # SelfHealingService
        mock_healing_instance = MockHealing.return_value
        mock_healing_instance.start_monitoring = AsyncMock()

        # get_redis_client returns a client instance.
        # lifespan calls `redis_client = get_redis_client()` then `await redis_client.disconnect()`
        mock_redis_instance = MockRedisClient.return_value
        mock_redis_instance.disconnect = AsyncMock()
        # Ensure other methods used are also async compatible if needed
        mock_redis_instance.cache_get = AsyncMock()
        mock_redis_instance.cache_mget = AsyncMock()
        mock_redis_instance.send_to_agent = AsyncMock()
        mock_redis_instance.broadcast_to_swarm = AsyncMock()
        # _client.keys is called in one endpoint
        mock_redis_instance._client.keys = AsyncMock(return_value=[])

        # Mock state objects (these might be overwritten by lifespan, but good to have)
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        # Ensure that if lifespan uses these classes, they return our configured instances
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = AsyncMock()
        app.state.self_healing = mock_healing_instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
