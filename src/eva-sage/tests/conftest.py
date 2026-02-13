import sys
from unittest.mock import MagicMock

# Mock shared module to avoid installing heavy dependencies
# This must be done before any import of eva_sage.main
if "shared" not in sys.modules:
    shared_mock = MagicMock()
    sys.modules["shared"] = shared_mock

    # Configure shared.get_settings
    settings_mock = MagicMock()
    shared_mock.get_settings.return_value = settings_mock

if "shared.redis_client" not in sys.modules:
    redis_client_mock = MagicMock()
    sys.modules["shared.redis_client"] = redis_client_mock

    # Configure redis functions
    redis_client_mock.init_redis = MagicMock()
    redis_client_mock.get_redis_client = MagicMock()
