import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch_geometric"] = MagicMock()
sys.modules["pandas"] = MagicMock()
