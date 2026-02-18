import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import types
import os
import json
import asyncio

# Define mocks similar to test_intent_router.py
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

# Mock shared module components
class MockIntentType:
    TRADING_ORDER = "TRADING_ORDER"
    POSITION_STATUS = "POSITION_STATUS"
    RISK_INQUIRY = "RISK_INQUIRY"
    MEMORY_RECALL = "MEMORY_RECALL"
    OSINT_REQUEST = "OSINT_REQUEST"
    SECURITY_ALERT = "SECURITY_ALERT"
    SYSTEM_COMMAND = "SYSTEM_COMMAND"
    GENERAL_CHAT = "GENERAL_CHAT"
    CHAT = "CHAT"

    def __new__(cls, value):
        valid_values = [
            "TRADING_ORDER", "POSITION_STATUS", "RISK_INQUIRY",
            "MEMORY_RECALL", "OSINT_REQUEST", "SECURITY_ALERT",
            "SYSTEM_COMMAND", "GENERAL_CHAT", "CHAT"
        ]
        if value in valid_values:
            return value
        raise ValueError(f"{value} is not a valid IntentType")

class MockIntent:
    def __init__(self, intent_type, target_expert, confidence, entities=None, **kwargs):
        self.intent_type = intent_type
        self.target_expert = target_expert
        self.confidence = confidence
        self.entities = entities or {}
        self.__dict__.update(kwargs)

    def __eq__(self, other):
        if not isinstance(other, MockIntent):
            return False
        return (
            self.intent_type == other.intent_type and
            self.target_expert == other.target_expert and
            self.confidence == other.confidence and
            self.entities == other.entities
        )

    def __repr__(self):
        return f"Intent(type={self.intent_type}, expert={self.target_expert}, conf={self.confidence})"

class MockMessageRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MockChatMessage:
    def __init__(self, role, content, **kwargs):
        self.role = role
        self.content = content
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"ChatMessage(role={self.role}, content={self.content})"

mock_shared = create_mock_module("shared")
mock_shared.IntentType = MockIntentType
mock_shared.Intent = MockIntent
mock_shared.MessageRole = MockMessageRole
mock_shared.ChatMessage = MockChatMessage
mock_modules["shared"] = mock_shared

# Mock eva_core.services.llm
# We need to mock get_llm_service function and the LLMService class/instance it returns
mock_llm_service_module = create_mock_module("eva_core.services.llm")
mock_llm_instance = MagicMock()
mock_llm_instance.generate_response = AsyncMock()

def get_llm_service_mock():
    return mock_llm_instance

mock_llm_service_module.get_llm_service = get_llm_service_mock
mock_modules["eva_core.services.llm"] = mock_llm_service_module


class TestStrategyOrchestrator(unittest.TestCase):
    def setUp(self):
        # Apply patch to sys.modules
        self.patcher = patch.dict(sys.modules, mock_modules)
        self.patcher.start()

        # Add source path to sys.path if not already there
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        if src_path not in sys.path:
            sys.path.append(src_path)

        # Ensure we can import the module under test
        import eva_core.strategy
        import importlib
        importlib.reload(eva_core.strategy)

        from eva_core.strategy import StrategyOrchestrator
        self.StrategyOrchestrator = StrategyOrchestrator
        self.IntentType = MockIntentType
        self.Intent = MockIntent
        self.ChatMessage = MockChatMessage
        self.MessageRole = MockMessageRole

        # Reset mock
        mock_llm_instance.generate_response.reset_mock()
        mock_llm_instance.generate_response.side_effect = None
        mock_llm_instance.generate_response.return_value = None
        self.mock_llm = mock_llm_instance

    def tearDown(self):
        self.patcher.stop()

    def test_route_request_success(self):
        # Create orchestrator
        orchestrator = self.StrategyOrchestrator()

        # Setup mock LLM response
        expected_json_str = json.dumps({
            "intent_type": "TRADING_ORDER",
            "target_expert": "banker",
            "confidence": 0.95,
            "entities": {"symbol": "XAUUSD", "action": "BUY"}
        })
        # The LLM returns a tuple (response_text, thoughts)
        self.mock_llm.generate_response.return_value = (expected_json_str, "Thinking process...")

        # Run method
        intent = asyncio.run(orchestrator.route_request("Buy gold"))

        # Verify result
        self.assertEqual(intent.intent_type, "TRADING_ORDER")
        self.assertEqual(intent.target_expert, "banker")
        self.assertEqual(intent.confidence, 0.95)
        self.assertEqual(intent.entities, {"symbol": "XAUUSD", "action": "BUY"})

        # Verify LLM called correctly
        self.mock_llm.generate_response.assert_called_once()
        call_args = self.mock_llm.generate_response.call_args
        kwargs = call_args.kwargs

        # Check messages argument
        messages = kwargs.get('messages')
        if not messages and len(call_args.args) > 0:
            messages = call_args.args[0]

        # Verify messages are ChatMessage objects
        self.assertTrue(isinstance(messages, list))
        self.assertTrue(len(messages) > 0)
        # We expect ChatMessage objects, not dicts
        # If the code is not fixed yet, this might fail or pass depending on what current code does
        # Current code uses dicts, so this assertion will fail if we check for ChatMessage
        # But this test defines the "Success" criteria for the FIXED code.
        self.assertTrue(hasattr(messages[0], 'role'), "Message should be an object with role attribute")
        self.assertEqual(messages[0].role, self.MessageRole.USER)
        self.assertEqual(messages[0].content, "Buy gold")

        # Verify json_mode is NOT passed (as it is not supported by LLMService signature)
        self.assertNotIn('json_mode', kwargs)

    def test_route_request_failure_json(self):
        orchestrator = self.StrategyOrchestrator()

        # Return invalid JSON
        self.mock_llm.generate_response.return_value = ("Not JSON", None)

        intent = asyncio.run(orchestrator.route_request("Hello"))

        # Should fallback to GENERAL_CHAT/core
        self.assertEqual(intent.intent_type, self.IntentType.GENERAL_CHAT)
        self.assertEqual(intent.target_expert, "core")
        self.assertEqual(intent.confidence, 0.1)

    def test_route_request_exception(self):
        orchestrator = self.StrategyOrchestrator()

        # Raise exception
        self.mock_llm.generate_response.side_effect = Exception("LLM Error")

        intent = asyncio.run(orchestrator.route_request("Hello"))

        self.assertEqual(intent.intent_type, self.IntentType.GENERAL_CHAT)
        self.assertEqual(intent.target_expert, "core")
        self.assertEqual(intent.confidence, 0.1)

if __name__ == '__main__':
    unittest.main()
