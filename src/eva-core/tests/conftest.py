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
    token = InternalAuth.generate_token("test-service")
    return {"X-Hive-Internal-Token": token}

@pytest.fixture
def client():
    # Patch dependencies in lifespan or global scope
    # Note: We need to configure the mock instance for EVAMQTTClient because lifespan calls await connect()
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client") as mock_get_redis, \
         patch("eva_core.main.EVAMQTTClient") as mock_mqtt_cls, \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService") as mock_self_healing_cls, \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        # Configure the mock instances
        mock_mqtt_instance = mock_mqtt_cls.return_value
        mock_mqtt_instance.connect = AsyncMock()

        mock_self_healing_instance = mock_self_healing_cls.return_value
        mock_self_healing_instance.start_monitoring = AsyncMock()

        mock_redis_client = mock_get_redis.return_value
        mock_redis_client.disconnect = AsyncMock()

        from eva_core.main import app
        # Mock state objects
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.mqtt = mock_mqtt_instance
        app.state.strategy_orchestrator = AsyncMock()
        app.state.self_healing = mock_self_healing_instance
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
