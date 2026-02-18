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
    # Prepare explicit mock instances
    mock_redis_client = MagicMock()
    mock_redis_client.disconnect = AsyncMock()

    mock_mqtt_client = MagicMock()
    mock_mqtt_client.connect = AsyncMock()

    mock_self_healing = MagicMock()
    mock_self_healing.start_monitoring = AsyncMock()

    # Patch dependencies using return_value for consistency
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis_client), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt_client), \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", return_value=mock_self_healing), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock), \
         patch("shared.internal_auth.InternalAuth.verify_token", return_value={"src": "test_agent"}):

        from eva_core.main import app
        # Mock state objects
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.mqtt = mock_mqtt_client
        app.state.strategy_orchestrator = AsyncMock()
        app.state.self_healing = mock_self_healing
        app.state.system_monitor = MagicMock()

        with TestClient(app) as c:
            yield c
