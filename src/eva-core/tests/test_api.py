"""
Configuration de tests pour eva-core
"""

import pytest
from fastapi.testclient import TestClient
from eva_core.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    """Vérifie que l'endpoint /health répond"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "core"

def test_agent_status_endpoint(client):
    """Vérifie l'endpoint status (simulation)"""
    response = client.get("/api/agents/status")
    assert response.status_code == 200
    assert "core" in response.json()
