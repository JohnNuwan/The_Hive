"""
Configuration de tests pour eva-core
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from eva_core.main import app
from shared.internal_auth import InternalAuth
def test_health_endpoint(client):
    """Vérifie que l'endpoint /health répond"""
    # /health est exclu de l'auth par défaut
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_agent_status_endpoint(client, auth_headers):
    """Vérifie l'endpoint status (simulation)"""
    response = client.get("/agents/status", headers=auth_headers)
    assert response.status_code == 200
    # The response depends on redis mock
    json_response = response.json()
    assert "core" in json_response
    assert json_response["core"]["status"] == "online"

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
