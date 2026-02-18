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
    # Pre-configure mock instances
    mock_mqtt_instance = MagicMock()
    mock_mqtt_instance.connect = AsyncMock()
    mock_mqtt_instance.publish = AsyncMock()

    mock_healing_instance = MagicMock()
    mock_healing_instance.start_monitoring = AsyncMock()
    mock_healing_instance.circuit_breaker = None

    mock_redis_instance = MagicMock()
    mock_redis_instance.disconnect = AsyncMock()
    mock_redis_instance.cache_get = AsyncMock(return_value={"status": "online", "ts": 1234567890})
    mock_redis_instance.cache_mget = AsyncMock(return_value=[{"status": "online", "ts": 1234567890}] * 7)
    mock_redis_instance.send_to_agent = AsyncMock()
    mock_redis_instance.broadcast_to_swarm = AsyncMock()
    mock_redis_instance._client.keys = AsyncMock(return_value=[])

    # Patch dependencies
    # For classes (EVAMQTTClient, SelfHealingService), return_value controls instantiation result
    # For functions (get_redis_client), return_value controls call result
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis_instance), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt_instance), \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", return_value=mock_healing_instance), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app

        # Manually assign mocked instances to app.state just in case lifespan doesn't run or is bypassed
        # But lifespan WILL run and overwrite these with the patched return values (which are the same objects)
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = AsyncMock()
        app.state.self_healing = mock_healing_instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
