import os
import pytest

@pytest.fixture(autouse=True)
def setup_env():
    os.environ["INTERNAL_SECRET_KEY"] = "test-internal-secret"
    os.environ["JWT_SECRET_KEY"] = "test-jwt-secret"
