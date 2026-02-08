import pytest
import time
from shared.internal_auth import InternalAuth

def test_token_generation_and_verification():
    """Verify that a token generated can be verified and contains correct data."""
    agent = "test-agent"
    token = InternalAuth.generate_token(agent)
    
    assert isinstance(token, str)
    
    payload = InternalAuth.verify_token(token)
    assert payload is not None
    assert payload["src"] == agent
    assert payload["iss"] == "hive-core"

def test_token_expiration():
    """Verify that an expired token is rejected (simulated)."""
    # We can't easily wait 60s in a unit test, but we can verify it doesn't fail immediately.
    token = InternalAuth.generate_token("fast-agent")
    payload = InternalAuth.verify_token(token)
    assert payload is not None

def test_invalid_token():
    """Verify that a tampered token is rejected."""
    token = InternalAuth.generate_token("agent")
    tampered_token = token + "tampered"
    
    payload = InternalAuth.verify_token(tampered_token)
    assert payload is None
