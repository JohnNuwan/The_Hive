import sys
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE any import happens in tests
# This allows running lightweight tests (like notifier) without installing numpy/torch

if "numpy" not in sys.modules:
    numpy_mock = MagicMock()
    # Mock specific numpy functions if needed, but for import purposes a simple mock suffices
    sys.modules["numpy"] = numpy_mock

if "torch" not in sys.modules:
    torch_mock = MagicMock()
    sys.modules["torch"] = torch_mock

if "pandas" not in sys.modules:
    pandas_mock = MagicMock()
    sys.modules["pandas"] = pandas_mock

# Also mock shared.math_ops to prevent it from trying to use the mocked numpy/torch during import if it has side effects
if "shared.math_ops" not in sys.modules:
    math_ops_mock = MagicMock()
    sys.modules["shared.math_ops"] = math_ops_mock
    # Mock functions imported in shared/__init__.py
    math_ops_mock.symlog = MagicMock()
    math_ops_mock.inv_symlog = MagicMock()
    math_ops_mock.calculate_var = MagicMock()
    math_ops_mock.calculate_cvar = MagicMock()
