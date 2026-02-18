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
    # Use AsyncMock for MQTTClient to ensure async methods like connect() are awaitable
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", new_callable=MagicMock) as MockRedisClient, \
         patch("eva_core.main.EVAMQTTClient", new_callable=MagicMock) as MockMQTT, \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", new_callable=MagicMock) as MockSelfHealing, \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app

        # Configure the MockMQTT instance to be async-compatible
        mock_mqtt_instance = MockMQTT.return_value
        # Important: connect() must be awaitable
        mock_mqtt_instance.connect = AsyncMock()

        # Configure SelfHealingService instance
        mock_self_healing_instance = MockSelfHealing.return_value
        # start_monitoring is started as a task, so it must be awaitable
        mock_self_healing_instance.start_monitoring = AsyncMock()

        # Configure Redis Client disconnect method which is awaited during shutdown
        mock_redis_instance = MockRedisClient.return_value
        mock_redis_instance.disconnect = AsyncMock()

        # Patch app.state.mqtt before entering context just in case (though lifespan overwrites)
        app.state.mqtt = mock_mqtt_instance
        app.state.self_healing = mock_self_healing_instance

        # Ensure strategy_orchestrator matches the mock if used
        app.state.strategy_orchestrator = MagicMock()

        with TestClient(app) as c:
            yield c
