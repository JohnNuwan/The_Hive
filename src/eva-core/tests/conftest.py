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
    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", new_callable=MagicMock) as MockGetRedis, \
         patch("eva_core.main.EVAMQTTClient", new_callable=MagicMock) as MockMQTT, \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", new_callable=MagicMock) as MockSelfHealing, \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        # Configure MockMQTT instance to have async connect
        mock_mqtt_instance = MockMQTT.return_value
        mock_mqtt_instance.connect = AsyncMock()

        # Configure MockSelfHealing instance to have async start_monitoring
        mock_self_healing_instance = MockSelfHealing.return_value
        mock_self_healing_instance.start_monitoring = AsyncMock()

        # Configure Redis Client to have async disconnect
        mock_redis_client = MockGetRedis.return_value
        mock_redis_client.disconnect = AsyncMock()

        from eva_core.main import app

        with TestClient(app) as c:
            # Re-apply mocks to app.state
            c.app.state.settings = MagicMock()
            c.app.state.intent_router = MagicMock()
            c.app.state.llm_service = MagicMock()
            c.app.state.memory_service = MagicMock()
            c.app.state.strategy_orchestrator = AsyncMock()
            c.app.state.self_healing = mock_self_healing_instance
            c.app.state.system_monitor = MagicMock()

            yield c
