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
    return {
        "Authorization": "Bearer test_token",
        "X-Hive-Internal-Token": "test_internal_token"
    }

@pytest.fixture
def client():
    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", new_callable=MagicMock) as MockRedis, \
         patch("eva_core.main.EVAMQTTClient", new_callable=MagicMock) as MockMQTT, \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", new_callable=MagicMock) as MockSelfHealing, \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock), \
         patch("shared.internal_auth.InternalAuth.verify_token", return_value={"src": "test_agent"}): # Mock token verification

        # Configure mocks to return AsyncMock for async methods

        # Redis Client setup
        redis_instance = MockRedis.return_value
        redis_instance.disconnect = AsyncMock()

        # MQTT Client setup
        mqtt_instance = MockMQTT.return_value
        mqtt_instance.connect = AsyncMock()

        # Self Healing Service setup (for asyncio.create_task)
        self_healing_instance = MockSelfHealing.return_value
        self_healing_instance.start_monitoring = AsyncMock()

        from eva_core.main import app
        # Mock state objects
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.mqtt = mqtt_instance
        app.state.strategy_orchestrator = AsyncMock()
        app.state.self_healing = self_healing_instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
