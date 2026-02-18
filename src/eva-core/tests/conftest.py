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
def client():
    # Setup mocks similar to test_api.py which is known to work

    # Redis Client: AsyncMock ensures all methods (disconnect, cache_get) are awaitable
    mock_redis_client_instance = AsyncMock()

    # MQTT Client: AsyncMock ensures connect() is awaitable
    mock_mqtt_instance = AsyncMock()
    mock_mqtt_instance.connect = AsyncMock()

    # SelfHealingService: start_monitoring is awaited/tasked
    mock_self_healing_instance = MagicMock()
    mock_self_healing_instance.start_monitoring = AsyncMock()

    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis_client_instance), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt_instance), \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", return_value=mock_self_healing_instance), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app

        # Mock state objects
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.strategy_orchestrator = AsyncMock()
        app.state.system_monitor = MagicMock()

        # The lifespan will overwrite app.state.mqtt and app.state.self_healing
        # with our pre-configured mocks

        with TestClient(app) as c:
            yield c
