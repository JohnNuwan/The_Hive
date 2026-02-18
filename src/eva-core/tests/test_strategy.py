
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import types
import os

# Define mocks similar to other tests
def create_mock_module(name):
    m = MagicMock(spec=types.ModuleType)
    m.__name__ = name
    return m

# Mock heavy dependencies
mock_modules = {
    "numpy": create_mock_module("numpy"),
    "torch": create_mock_module("torch"),
    "fastapi": create_mock_module("fastapi"),
    "redis": create_mock_module("redis"),
    "paho": create_mock_module("paho"),
    "paho.mqtt": create_mock_module("paho.mqtt"),
    "jwt": create_mock_module("jwt"),
    "neo4j": create_mock_module("neo4j"),
    "mem0ai": create_mock_module("mem0ai"),
    "qdrant_client": create_mock_module("qdrant_client"),
    "langchain_ollama": create_mock_module("langchain_ollama"),
    "langchain_core": create_mock_module("langchain_core"),
    "eva_core.main": create_mock_module("eva_core.main"),
    "eva_core.services": create_mock_module("eva_core.services"),
    "eva_core.services.council": create_mock_module("eva_core.services.council"),
    "eva_core.services.llm": create_mock_module("eva_core.services.llm"),
    "shared.memory_bridge": create_mock_module("shared.memory_bridge"),
}
mock_modules["eva_core.main"].app = MagicMock()
# Link the submodule to the parent package mock
mock_modules["eva_core.services"].llm = mock_modules["eva_core.services.llm"]

# Define get_llm_service on the mock module so patch can find it
mock_modules["eva_core.services.llm"].get_llm_service = MagicMock()

# Mock shared module
mock_shared = create_mock_module("shared")

class MockIntentType:
    CHAT = "CHAT"
    TRADE = "TRADE"
    def __call__(self, value):
        return value
    def __new__(cls, value):
        return value

class MockIntent:
    def __init__(self, intent_type, target_expert, confidence, entities=None):
        self.intent_type = intent_type
        self.target_expert = target_expert
        self.confidence = confidence
        self.entities = entities or {}

class MockChatMessage:
    def __init__(self, **kwargs):
        pass

class MockMessageRole:
    USER = "user"

mock_shared.IntentType = MockIntentType
mock_shared.Intent = MockIntent
mock_shared.ChatMessage = MockChatMessage
mock_shared.MessageRole = MockMessageRole
mock_modules["shared"] = mock_shared

class TestStrategyOrchestrator(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict(sys.modules, mock_modules)
        self.patcher.start()

        # Add source path
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        if src_path not in sys.path:
            sys.path.append(src_path)

        # Ensure imports work for patching
        import eva_core.services.llm

        # Mock get_llm_service
        self.mock_llm_service = MagicMock()
        self.mock_llm_service.generate_response = AsyncMock()

        # Patch get_llm_service in the module
        self.llm_patcher = patch("eva_core.services.llm.get_llm_service", return_value=self.mock_llm_service)
        self.llm_patcher.start()

        import eva_core.strategy
        import importlib
        importlib.reload(eva_core.strategy)
        self.strategy_module = eva_core.strategy

    def tearDown(self):
        self.llm_patcher.stop()
        self.patcher.stop()

    def test_route_request_success(self):
        orchestrator = self.strategy_module.StrategyOrchestrator()

        # Mock successful response
        # generate_response returns tuple (response_text, thoughts)
        response_json = '{"intent_type": "CHAT", "target_expert": "core", "confidence": 0.9}'
        self.mock_llm_service.generate_response.return_value = (response_json, None)

        async def run():
            return await orchestrator.route_request("Hello world")

        result = asyncio.run(run())

        self.assertEqual(result.intent_type, "CHAT")
        self.assertEqual(result.target_expert, "core")
        self.assertEqual(result.confidence, 0.9)

    def test_route_request_failure_fallback(self):
        orchestrator = self.strategy_module.StrategyOrchestrator()

        # Mock failure (e.g. invalid json)
        self.mock_llm_service.generate_response.return_value = ("invalid json", None)

        async def run():
            return await orchestrator.route_request("Hello world")

        result = asyncio.run(run())

        self.assertEqual(result.intent_type, "CHAT")
        self.assertEqual(result.target_expert, "core")
        self.assertEqual(result.confidence, 0.1)

if __name__ == '__main__':
    unittest.main()
