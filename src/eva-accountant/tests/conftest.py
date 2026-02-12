import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules["numpy"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["pandas"] = MagicMock()
