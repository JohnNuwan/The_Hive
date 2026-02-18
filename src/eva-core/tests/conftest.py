import sys
from unittest.mock import MagicMock, AsyncMock, patch

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

@pytest.fixture
def auth_headers():
    return {"X-Hive-Internal-Token": "test-token"}

@pytest.fixture
def client():
    # Setup mocks for Redis client
    mock_redis_client = MagicMock()
    mock_redis_client.disconnect = AsyncMock()

    # Setup mocks for MQTT
    mock_mqtt = AsyncMock()
    mock_mqtt.connect = AsyncMock()

    # Setup mocks for SelfHealing
    mock_healing = AsyncMock()
    mock_healing.start_monitoring = AsyncMock()

    # Patch dependencies in lifespan or global scope
    with patch("eva_core.main.init_redis", new_callable=AsyncMock), \
         patch("eva_core.main.get_redis_client", return_value=mock_redis_client), \
         patch("eva_core.main.EVAMQTTClient", return_value=mock_mqtt), \
         patch("eva_core.main.StrategyOrchestrator", new_callable=MagicMock), \
         patch("eva_core.main.SelfHealingService", return_value=mock_healing), \
         patch("eva_core.services.llm.LLMService", new_callable=MagicMock), \
         patch("eva_core.services.memory.MemoryService", new_callable=MagicMock):

        from eva_core.main import app

        # Ensure mocked instances are attached to app.state
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()

        # Critical async mocks
        app.state.mqtt = mock_mqtt
        app.state.self_healing = mock_healing
        app.state.strategy_orchestrator = AsyncMock()

        app.state.system_monitor = MagicMock()

        # IMPORTANT: Mock PromptMaster to prevent it from loading files
        app.state.prompt_master = MagicMock()

        # Mock the imported classes to return async instances where needed
        # Note: StrategyOrchestrator and SelfHealingService are already patched in the context above
        # but we need to ensure any re-imports inside main or other modules also get the mocks if they instantiate

        # FIX: TestClient with async lifespan needs careful handling.
        # Using context manager is correct, but ensure no exceptions in lifespan.
        with TestClient(app) as c:
            yield c
