import pytest
from unittest.mock import MagicMock
from eva_core.main import app

def test_circuit_breaker_status_nominal(client, auth_headers):
    """
    Test the circuit breaker status when no circuit breaker is attached.
    Expects fallback/default status.
    """
    # Ensure self_healing.circuit_breaker is None
    # app.state.self_healing is a MagicMock from conftest.py
    app.state.self_healing.circuit_breaker = None

    response = client.get("/circuit-breaker/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "core_circuit_breaker"
    assert data["state"] == "CLOSED"
    assert data["failures"] == 0
    assert data["failure_threshold"] == 5

def test_circuit_breaker_status_active(client, auth_headers):
    """
    Test the circuit breaker status when a circuit breaker is active.
    Expects the status from the circuit breaker.
    """
    # Mock the circuit breaker on self_healing service
    mock_cb = MagicMock()
    mock_cb.get_status.return_value = {
        "name": "core_circuit_breaker",
        "state": "OPEN",
        "failures": 10,
        "failure_threshold": 5,
        "last_failure": "Timeout"
    }

    app.state.self_healing.circuit_breaker = mock_cb

    response = client.get("/circuit-breaker/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["state"] == "OPEN"
    assert data["failures"] == 10
    assert data["last_failure"] == "Timeout"
