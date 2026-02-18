
import pytest
from unittest.mock import MagicMock

def test_circuit_breaker_status_nominal(client):
    pass

@pytest.mark.asyncio
async def test_circuit_breaker_status_nominal(client):
    headers = {"X-Hive-Internal-Token": "test-token"}

    # Force circuit_breaker to be None (not a Mock) to trigger the fallback path
    # which returns a simple dict, not a mock object
    client.app.state.self_healing.circuit_breaker = None

    with pytest.MonkeyPatch.context() as m:
        m.setattr("shared.internal_auth.InternalAuth.verify_token", lambda x: {"src": "test", "role": "internal"})

        response = client.get("/circuit-breaker/status", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "CLOSED"

@pytest.mark.asyncio
async def test_circuit_breaker_status_active(client):
    headers = {"X-Hive-Internal-Token": "test-token"}

    # Setup circuit breaker mock
    mock_cb = MagicMock()
    mock_cb.get_status.return_value = {
        "name": "core_circuit_breaker",
        "state": "OPEN",
        "failures": 10,
        "failure_threshold": 5
    }
    client.app.state.self_healing.circuit_breaker = mock_cb

    with pytest.MonkeyPatch.context() as m:
        m.setattr("shared.internal_auth.InternalAuth.verify_token", lambda x: {"src": "test", "role": "internal"})

        response = client.get("/circuit-breaker/status", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "OPEN"
