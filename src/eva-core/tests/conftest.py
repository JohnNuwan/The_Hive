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
from contextlib import asynccontextmanager
from shared.internal_auth import InternalAuth

class FakeMQTT:
    def __init__(self, *args, **kwargs): pass
    async def connect(self): pass
    async def publish(self, *args, **kwargs): pass

class FakeStrategyOrchestrator:
    def __init__(self, *args, **kwargs): pass
    async def route_request(self, *args, **kwargs): pass

class FakeSelfHealingService:
    def __init__(self, *args, **kwargs): pass
    async def start_monitoring(self): pass

class FakeRedis:
    async def disconnect(self): pass
    async def cache_mget(self, keys): return [None] * len(keys)
    @property
    def _client(self):
        m = MagicMock()
        m.keys = AsyncMock(return_value=[])
        return m

@pytest.fixture
def client():
    from eva_core.main import app

    # Override lifespan to avoid complex patching of startup logic
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def mock_lifespan(app):
        # Initialize mock state
        app.state.settings = MagicMock()
        app.state.intent_router = MagicMock()
        app.state.llm_service = MagicMock()
        app.state.memory_service = MagicMock()
        app.state.prompt_master = MagicMock()

        # Use Fakes for async services
        app.state.mqtt = FakeMQTT()
        app.state.strategy_orchestrator = FakeStrategyOrchestrator()
        app.state.self_healing = FakeSelfHealingService()
        app.state.system_monitor = MagicMock()

        # Telemetry
        from datetime import datetime
        app.state.start_time = datetime.now()
        app.state.request_count = 0
        app.state.error_count = 0

        yield

    app.router.lifespan_context = mock_lifespan

    fake_redis = FakeRedis()

    with patch("eva_core.main.get_redis_client", return_value=fake_redis), \
         patch("shared.redis_client.get_redis_client", return_value=fake_redis):

        with TestClient(app) as c:
            yield c

    app.router.lifespan_context = original_lifespan

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-core")
    return {"X-Hive-Internal-Token": token}
