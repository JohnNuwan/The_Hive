import sys
from unittest.mock import MagicMock, patch

# Mock eva_core.main to prevent full app startup
# This must happen before any import from eva_core
if "eva_core.main" not in sys.modules:
    sys.modules["eva_core.main"] = MagicMock()

import pytest

def test_memory_service_import():
    """Verify that MemoryService can be imported and instantiated."""
    # We need to ensure eva_core.main is mocked
    # Note: If tests run in parallel or shared process, this might affect others,
    # but for this specific verify step it is fine.

    from eva_core.services.memory import MemoryService

    # Mock settings
    with patch("eva_core.services.memory.get_settings") as mock_settings:
        mock_settings.return_value.qdrant_host = "localhost"
        mock_settings.return_value.qdrant_port = 6333
        mock_settings.return_value.qdrant_collection_conversations = "test_collection"

        # Instantiate
        service = MemoryService()
        assert service is not None
        assert service.host == "localhost"
