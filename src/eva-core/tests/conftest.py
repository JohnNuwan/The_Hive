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
import shared.redis_client

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}

@pytest.fixture
def client():
    # Setup mocks ensuring async compatibility
    mock_redis_client = AsyncMock()
    # Explicitly set async methods to avoid ambiguity
    mock_redis_client.get = AsyncMock(return_value=None)
    mock_redis_client.set = AsyncMock(return_value=True)
    mock_redis_client.cache_get = AsyncMock(return_value={"status": "online", "ts": 1234567890})
    mock_redis_client.cache_mget = AsyncMock(return_value=[{"status": "online", "ts": 1234567890}] * 7)

    # Mock underlying redis client for raw access
    mock_redis_client._client = AsyncMock()
    mock_redis_client._client.keys = AsyncMock(return_value=[])
    mock_redis_client._client.mget = AsyncMock(return_value=[])

    mock_mqtt_instance = MagicMock()
    mock_mqtt_instance.connect = AsyncMock()

    mock_self_healing_instance = MagicMock()
    # Create a real coroutine for start_monitoring to satisfy asyncio.create_task
    async def mock_start_monitoring():
        pass
    mock_self_healing_instance.start_monitoring = MagicMock(side_effect=mock_start_monitoring)

    mock_memory_service = MagicMock()
    mock_memory_service.store_message = AsyncMock()

    # Patch dependencies in lifespan or global scope
    # We patch shared.redis_client._redis_client singleton AND get_redis_client function
    # AND init_redis to ensure it doesn't try to connect real redis
    with patch.object(shared.redis_client, "_redis_client", mock_redis_client), \
         patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis_client), \
         patch("shared.redis_client.get_redis_client", return_value=mock_redis_client), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt_instance), \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", return_value=mock_self_healing_instance), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", return_value=mock_memory_service):

        from eva_core.main import app
        # Mock state objects
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = mock_memory_service
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = AsyncMock()
        app.state.self_healing = mock_self_healing_instance
        app.state.system_monitor = MagicMock()

        # Initialize manual mocks for internal components if needed
        # app.state.prompt_master is initialized in lifespan, we let it be or mock it?
        # Lifespan will overwrite app.state. So we rely on patches.

        with TestClient(app) as c:
            yield c
