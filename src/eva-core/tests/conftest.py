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

# Redis (if not present)
if "redis" not in sys.modules:
    sys.modules["redis"] = MagicMock()
if "redis.asyncio" not in sys.modules:
    sys.modules["redis.asyncio"] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Need to import InternalAuth for token generation, but ensure shared is importable
# Assuming shared is available via PYTHONPATH or installed
try:
    from shared.internal_auth import InternalAuth
except ImportError:
    InternalAuth = MagicMock()
    InternalAuth.generate_token.return_value = "mock-token"

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}

@pytest.fixture
def client():
    # Helper to mock class instances with async methods
    def mock_mqtt_client(*args, **kwargs):
        m = MagicMock()
        m.connect = AsyncMock()
        return m

    def mock_self_healing_service(*args, **kwargs):
        m = MagicMock()
        m.start_monitoring = AsyncMock()
        return m

    def mock_strategy_orchestrator(*args, **kwargs):
        m = MagicMock()
        m.route_request = AsyncMock()
        return m

    def mock_redis_client_func(*args, **kwargs):
        m = MagicMock()
        m.disconnect = AsyncMock()
        m.cache_get = AsyncMock()
        m.cache_set = AsyncMock()
        m.cache_mget = AsyncMock()
        m.broadcast_to_swarm = AsyncMock()
        m.send_to_agent = AsyncMock()
        m._client = MagicMock() # underlying client
        return m

    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", side_effect=mock_redis_client_func), \
         patch("eva_core.main.EVAMQTTClient", side_effect=mock_mqtt_client), \
         patch("eva_core.main.StrategyOrchestrator", side_effect=mock_strategy_orchestrator), \
         patch("eva_core.main.SelfHealingService", side_effect=mock_self_healing_service), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app
        # Mock state objects
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.mqtt = AsyncMock()
        app.state.strategy_orchestrator = AsyncMock()
        app.state.self_healing = AsyncMock()
        app.state.system_monitor = MagicMock()
        app.state.start_time = MagicMock() # avoid datetime.now issues if needed

        with TestClient(app) as c:
            yield c
