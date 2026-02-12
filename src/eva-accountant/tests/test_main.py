import pytest
from unittest.mock import mock_open, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import json
import os

# Import app and internal objects
from eva_accountant.main import app, financial_state, LEDGER_FILE

@pytest.fixture(autouse=True)
def reset_state():
    """Reset financial state before each test to ensure isolation."""
    # Save original state
    original_state = financial_state.copy()
    original_expenses = list(financial_state["expenses_detail"]) # Deep copy for list

    # Reset to default
    financial_state["gross_profit"] = 0.0
    financial_state["tax_provision"] = 0.0
    financial_state["operating_expenses"] = 0.0
    financial_state["net_roi"] = 0.0
    financial_state["expenses_detail"] = []

    yield

    # Restore original state
    financial_state.update(original_state)
    financial_state["expenses_detail"] = original_expenses

@pytest.fixture
def client(mock_redis):
    """Create a TestClient with mocked background tasks and persistence."""
    # Patch external interactions and long-running tasks
    with patch("eva_accountant.main.init_redis", new_callable=AsyncMock), \
         patch("eva_accountant.main.get_redis_client", return_value=mock_redis), \
         patch("eva_accountant.main.hard_heartbeat", new_callable=AsyncMock), \
         patch("eva_accountant.main.load_ledger"), \
         patch("eva_accountant.main.save_ledger"): # Disable save on shutdown to avoid file I/O

        with TestClient(app) as c:
            yield c

def test_save_ledger_writes_json():
    """Test that save_ledger writes the financial state to a JSON file."""
    from eva_accountant.main import save_ledger

    # Set some state to verify it's written
    financial_state["gross_profit"] = 1234.56

    m = mock_open()
    with patch("builtins.open", m):
        save_ledger()

    m.assert_called_with(LEDGER_FILE, "w", encoding="utf-8")

    # Verify content
    handle = m()
    written_data = "".join(call.args[0] for call in handle.write.mock_calls)
    data = json.loads(written_data)
    assert data["gross_profit"] == 1234.56
    assert "expenses_detail" in data

def test_load_ledger_restores_state():
    """Test that load_ledger updates financial state from file."""
    from eva_accountant.main import load_ledger

    fake_data = json.dumps({
        "gross_profit": 999.99,
        "operating_expenses": 100.0,
        "expenses_detail": [{"desc": "loaded expense"}]
    })

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=fake_data)):
        load_ledger()

    assert financial_state["gross_profit"] == 999.99
    assert financial_state["operating_expenses"] == 100.0
    assert len(financial_state["expenses_detail"]) == 1
    assert financial_state["expenses_detail"][0]["desc"] == "loaded expense"

def test_load_ledger_handles_missing_file():
    """Test that load_ledger gracefully handles missing file."""
    from eva_accountant.main import load_ledger

    # Ensure state is clean (handled by fixture, but good to be explicit)
    assert financial_state["gross_profit"] == 0.0

    with patch("os.path.exists", return_value=False):
        load_ledger()

    # State should remain default
    assert financial_state["gross_profit"] == 0.0

def test_load_ledger_handles_error():
    """Test that load_ledger logs error but doesn't crash on bad file."""
    from eva_accountant.main import load_ledger

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", side_effect=IOError("Permission denied")):
        load_ledger()

    # Should just log error and keep going (state remains default)
    assert financial_state["gross_profit"] == 0.0

def test_health_endpoint(client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_report_endpoint(client, auth_headers):
    """Test /report endpoint returns current state."""
    financial_state["net_roi"] = 42.0
    response = client.get("/report", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["net"] == 42.0

def test_expense_flow(client, auth_headers):
    """Test registering an expense updates state and saves ledger."""
    payload = {
        "description": "Cloud hosting",
        "amount": 50.0,
        "category": "infrastructure"
    }

    # We patch save_ledger specifically to verify it's called during the request
    with patch("eva_accountant.main.save_ledger") as mock_save:
        response = client.post("/expense", json=payload, headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["status"] == "recorded"

    # Verify state update
    assert financial_state["operating_expenses"] == 50.0
    # Net ROI = 0 - 0 - 50 = -50
    assert financial_state["net_roi"] == -50.0

    # Verify save was called
    mock_save.assert_called_once()

def test_sync_ledger_flow(client, auth_headers):
    """Test syncing with compliance updates state and saves ledger."""
    payload = {
        "total_profit": 5000.0,
        "total_tax": 1000.0
    }

    # Initial expense to verify net roi calc
    financial_state["operating_expenses"] = 200.0

    with patch("eva_accountant.main.save_ledger") as mock_save:
        response = client.post("/sync-ledger", json=payload, headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["status"] == "synchronized"

    # Verify state
    assert financial_state["gross_profit"] == 5000.0
    assert financial_state["tax_provision"] == 1000.0
    # Net ROI = 5000 - 1000 - 200 = 3800
    assert financial_state["net_roi"] == 3800.0

    # Verify save was called
    mock_save.assert_called_once()

def test_unauthorized_access(client):
    """Test that protected endpoints require authentication."""
    from fastapi import HTTPException
    # Middleware raises HTTPException which propagates because it's outside the exception handler scope in this setup
    with pytest.raises(HTTPException) as excinfo:
        client.post("/expense", json={})
    assert excinfo.value.status_code == 401
