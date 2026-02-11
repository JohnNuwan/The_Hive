import sys
import os
import time
import re
from unittest.mock import MagicMock
import types

# Helper to create a mock module
def create_mock_module(name):
    m = MagicMock(spec=types.ModuleType)
    m.__name__ = name
    m.__path__ = []
    m.__spec__ = MagicMock()
    return m

# Mock heavy dependencies
modules_to_mock = [
    "numpy", "torch", "fastapi", "fastapi.middleware", "fastapi.middleware.cors",
    "starlette", "starlette.middleware", "starlette.middleware.base",
    "pydantic", "pydantic_settings", "redis", "langchain_ollama",
    "langchain_core", "langchain_core.prompts", "psutil", "paho", "paho.mqtt",
    "jwt", "neo4j", "mem0ai", "qdrant_client"
]

for mod_name in modules_to_mock:
    sys.modules[mod_name] = create_mock_module(mod_name)

# Mock eva_core.main to prevent importing the whole app
mock_main = create_mock_module("eva_core.main")
mock_main.app = MagicMock()
sys.modules["eva_core.main"] = mock_main

# Mock shared module
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
        return self.__dict__ == other.__dict__

mock_shared = create_mock_module("shared")
mock_shared.IntentType = MockIntentType
mock_shared.Intent = MockIntent
sys.modules["shared"] = mock_shared

# Ensure src directories are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/eva-core')))

# Now import the module under test
try:
    from eva_core.router.intent import IntentRouter
except ImportError as e:
    print(f"Error importing IntentRouter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def benchmark():
    # Force use_llm=False to use regex pattern matching
    router = IntentRouter(use_llm=False)

    test_cases = [
        "achète 1 lot de gold avec un sl à 2000",
        "quel est le statut de mes positions ouvertes ?",
        "analyse le risque actuel du portfolio",
        "rappelle-moi ce qu'on a fait hier",
        "trouve des infos sur cette adresse IP",
        "alerte intrusion détectée sur le serveur",
        "redémarre le système de trading",
        "bonjour comment ça va today ?", # General chat
        "buy 0.5 lots eur/usd sl 1.05 tp 1.10",
        "what is my current drawdown?",
    ]

    # Warm up
    for text in test_cases:
        router._classify_with_patterns(text)

    iterations = 50000
    start_time = time.time()

    for _ in range(iterations):
        for text in test_cases:
            router._classify_with_patterns(text)

    end_time = time.time()
    total_time = end_time - start_time
    ops_per_sec = (iterations * len(test_cases)) / total_time

    print(f"Total time: {total_time:.4f} seconds")
    print(f"Operations per second: {ops_per_sec:.2f}")

if __name__ == "__main__":
    benchmark()
