import sys
from unittest.mock import MagicMock

# Mock heavy dependencies that are not needed for unit tests
# This prevents ImportErrors when running in a lightweight environment

mock_torch = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.nn.functional = MagicMock()
# Mock is_tensor to allow basic checks
mock_torch.is_tensor = MagicMock(return_value=False)

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.nn.functional"] = mock_torch.nn.functional
sys.modules["torch_geometric"] = MagicMock()
sys.modules["neo4j"] = MagicMock()
sys.modules["langchain"] = MagicMock()
sys.modules["qdrant_client"] = MagicMock()
sys.modules["mem0ai"] = MagicMock()
sys.modules["mem0"] = MagicMock()
sys.modules["pandas"] = MagicMock()
