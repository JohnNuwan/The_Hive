
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Define Mocks
mock_shared = MagicMock()

# Mock IntentType as a class with attributes
class MockIntentType:
    CHAT = "CHAT"
    TRADE = "TRADE"
    def __new__(cls, value):
        # Allow instantiation with a value if needed, or just return value
        return value

class MockIntent:
    def __init__(self, intent_type, target_expert, confidence, entities=None):
        self.intent_type = intent_type
        self.target_expert = target_expert
        self.confidence = confidence
        self.entities = entities or {}

class MockChatMessage:
    def __init__(self, session_id, role, content, thoughts=None):
        self.session_id = session_id
        self.role = role
        self.content = content
        self.thoughts = thoughts

class MockMessageRole:
    USER = "user"

mock_shared.Intent = MockIntent
mock_shared.IntentType = MockIntentType
mock_shared.ChatMessage = MockChatMessage
mock_shared.MessageRole = MockMessageRole
sys.modules["shared"] = mock_shared

# Mock eva_core.main
sys.modules["eva_core.main"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["pydantic"] = MagicMock()

# Mock eva_core.services.llm
mock_llm_service_mod = MagicMock()
sys.modules["eva_core.services.llm"] = mock_llm_service_mod

import os
src_path = os.path.abspath("src/eva-core")
if src_path not in sys.path:
    sys.path.append(src_path)
shared_path = os.path.abspath("src/shared")
if shared_path not in sys.path:
    sys.path.append(shared_path)

from eva_core.strategy import StrategyOrchestrator

class TestStrategyOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def test_route_request_success(self):
        # Setup the mock LLM service instance
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate_response = AsyncMock(return_value=('{"intent_type": "TRADE", "target_expert": "banker", "confidence": 0.9}', None))

        # Configure the get_llm_service mock to return our instance
        mock_llm_service_mod.get_llm_service.return_value = mock_llm_instance

        orchestrator = StrategyOrchestrator()

        intent = await orchestrator.route_request("buy gold")

        self.assertEqual(intent.intent_type, MockIntentType.TRADE)
        self.assertEqual(intent.target_expert, "banker")
        self.assertEqual(intent.confidence, 0.9)

        # Verify call args
        call_args = mock_llm_instance.generate_response.call_args
        kwargs = call_args.kwargs
        # Verify json_mode is NOT present (default is None/missing in kwargs if not passed)
        self.assertNotIn('json_mode', kwargs)

    async def test_route_request_failure_fallback(self):
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate_response = AsyncMock(side_effect=Exception("LLM Error"))
        mock_llm_service_mod.get_llm_service.return_value = mock_llm_instance

        orchestrator = StrategyOrchestrator()
        intent = await orchestrator.route_request("hello")

        # This will use IntentType.CHAT
        self.assertEqual(intent.intent_type, MockIntentType.CHAT)
        self.assertEqual(intent.target_expert, "core")
        self.assertEqual(intent.confidence, 0.1)

if __name__ == "__main__":
    unittest.main()
