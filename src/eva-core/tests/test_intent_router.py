import sys
import unittest
from unittest.mock import MagicMock, patch
import types
import os

# Define mocks similar to benchmark
class MockModule(MagicMock):
    @property
    def __path__(self):
        return []
    @property
    def __spec__(self):
        return MagicMock()

def create_mock_module(name):
    m = MagicMock(spec=types.ModuleType)
    m.__name__ = name
    m.__path__ = []
    m.__spec__ = MagicMock()
    return m

# Mock heavy dependencies
mock_modules = {
    "numpy": create_mock_module("numpy"),
    "torch": create_mock_module("torch"),
    "fastapi": create_mock_module("fastapi"),
    "fastapi.middleware": create_mock_module("fastapi.middleware"),
    "fastapi.middleware.cors": create_mock_module("fastapi.middleware.cors"),
    "starlette": create_mock_module("starlette"),
    "starlette.middleware": create_mock_module("starlette.middleware"),
    "starlette.middleware.base": create_mock_module("starlette.middleware.base"),
    "pydantic": create_mock_module("pydantic"),
    "pydantic_settings": create_mock_module("pydantic_settings"),
    "redis": create_mock_module("redis"),
    "langchain_ollama": create_mock_module("langchain_ollama"),
    "langchain_core": create_mock_module("langchain_core"),
    "langchain_core.prompts": create_mock_module("langchain_core.prompts"),
    "psutil": create_mock_module("psutil"),
    "paho": create_mock_module("paho"),
    "paho.mqtt": create_mock_module("paho.mqtt"),
    "jwt": create_mock_module("jwt"),
    "neo4j": create_mock_module("neo4j"),
    "mem0ai": create_mock_module("mem0ai"),
    "qdrant_client": create_mock_module("qdrant_client"),
    "eva_core.main": create_mock_module("eva_core.main"),
}
mock_modules["eva_core.main"].app = MagicMock()

# Mock shared
class MockIntentType:
    TRADING_ORDER = "TRADING_ORDER"
    POSITION_STATUS = "POSITION_STATUS"
    RISK_INQUIRY = "RISK_INQUIRY"
    MEMORY_RECALL = "MEMORY_RECALL"
    OSINT_REQUEST = "OSINT_REQUEST"
    SECURITY_ALERT = "SECURITY_ALERT"
    SYSTEM_COMMAND = "SYSTEM_COMMAND"
    GENERAL_CHAT = "GENERAL_CHAT"

class MockIntent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __eq__(self, other):
        return self.intent_type == other.intent_type and self.entities == other.entities

mock_shared = create_mock_module("shared")
mock_shared.IntentType = MockIntentType
mock_shared.Intent = MockIntent
mock_modules["shared"] = mock_shared

class TestIntentRouter(unittest.TestCase):
    def setUp(self):
        # Apply patch to sys.modules
        self.patcher = patch.dict(sys.modules, mock_modules)
        self.patcher.start()

        # Add source path to sys.path if not already there
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        if src_path not in sys.path:
            sys.path.append(src_path)

        # Import inside test to ensure mocks are used
        # We need to reload if it was already imported
        import eva_core.router.intent
        import importlib
        importlib.reload(eva_core.router.intent)

        from eva_core.router.intent import IntentRouter
        self.IntentRouter = IntentRouter
        self.IntentType = MockIntentType

    def tearDown(self):
        self.patcher.stop()

    def test_classify_trading_order(self):
        router = self.IntentRouter(use_llm=False)
        text = "achète 1 lot de gold avec un sl à 2000"
        intent = router._classify_with_patterns(text)
        self.assertEqual(intent.intent_type, self.IntentType.TRADING_ORDER)
        self.assertEqual(intent.entities.get("action"), "BUY")
        self.assertEqual(intent.entities.get("volume"), 1.0)
        self.assertEqual(intent.entities.get("symbol"), "XAUUSD")
        self.assertEqual(intent.entities.get("stop_loss"), 2000.0)

    def test_classify_position_status(self):
        router = self.IntentRouter(use_llm=False)
        text = "quel est le statut de mes positions ?"
        intent = router._classify_with_patterns(text)
        self.assertEqual(intent.intent_type, self.IntentType.POSITION_STATUS)

    def test_classify_risk_inquiry(self):
        router = self.IntentRouter(use_llm=False)
        text = "quel est le drawdown actuel ?"
        intent = router._classify_with_patterns(text)
        self.assertEqual(intent.intent_type, self.IntentType.RISK_INQUIRY)

    def test_classify_general_chat(self):
        router = self.IntentRouter(use_llm=False)
        text = "bonjour comment ça va ?"
        intent = router._classify_with_patterns(text)
        self.assertEqual(intent.intent_type, self.IntentType.GENERAL_CHAT)

    def test_extract_trading_entities_advanced(self):
        router = self.IntentRouter(use_llm=False)
        text = "buy 0.5 lots eur/usd sl 1.05 tp 1.10"
        intent = router._classify_with_patterns(text)
        self.assertEqual(intent.intent_type, self.IntentType.TRADING_ORDER)
        self.assertEqual(intent.entities.get("action"), "BUY")
        self.assertEqual(intent.entities.get("volume"), 0.5)
        self.assertEqual(intent.entities.get("symbol"), "EURUSD")
        self.assertEqual(intent.entities.get("stop_loss"), 1.05)
        self.assertEqual(intent.entities.get("take_profit"), 1.10)

if __name__ == '__main__':
    unittest.main()
