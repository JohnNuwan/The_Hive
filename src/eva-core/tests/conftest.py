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
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", new_callable=MagicMock) as MockGetRedis, \
         patch("eva_core.main.EVAMQTTClient") as MockMQTTClient, \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", new_callable=MagicMock) as MockSelfHealing, \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock), \
         patch("eva_core.main.SystemMonitor", new_callable=MagicMock):

        # Make Redis disconnect awaitable
        MockGetRedis.return_value.disconnect = AsyncMock()

        # Make MQTT connect awaitable
        MockMQTTClient.return_value.connect = AsyncMock()

        # Make SelfHealingService start_monitoring awaitable
        MockSelfHealing.return_value.start_monitoring = AsyncMock()

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

        with TestClient(app) as c:
            yield c
