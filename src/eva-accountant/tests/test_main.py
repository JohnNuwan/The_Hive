import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from eva_accountant.main import app, financial_state
from shared.internal_auth import InternalAuth

@pytest.fixture
def mock_redis():
    mock_client = AsyncMock()
    mock_client.cache_set = AsyncMock()
    return mock_client

@pytest.fixture
def client(mock_redis):
    # Mock init_redis and get_redis_client
    with patch("eva_accountant.main.init_redis", new_callable=AsyncMock), \
         patch("eva_accountant.main.get_redis_client", return_value=mock_redis), \
         patch("eva_accountant.main.hard_heartbeat", new_callable=AsyncMock), \
         patch("eva_accountant.main.load_ledger", new_callable=MagicMock), \
         patch("eva_accountant.main.save_ledger", new_callable=MagicMock):

        # Reset financial_state for each test
        financial_state["gross_profit"] = 0.0
        financial_state["tax_provision"] = 0.0
        financial_state["operating_expenses"] = 0.0
        financial_state["net_roi"] = 0.0
        financial_state["expenses_detail"] = []

        with TestClient(app) as c:
            yield c

@pytest.fixture
def auth_headers():
    token = InternalAuth.generate_token("test-accountant")
    return {"X-Hive-Internal-Token": token}

def test_register_expense_negative_amount(client, auth_headers):
    """
    Test registering an expense with a negative amount.
    """
    expense_data = {
        "description": "Refund or correction",
        "amount": -50.0,
        "category": "infrastructure"
    }

    # Initial state
    assert financial_state["operating_expenses"] == 0.0

    response = client.post("/expense", json=expense_data, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"

    # Verify financial state update
    # Negative expense means operating_expenses decreases (becomes negative if it was 0)
    assert financial_state["operating_expenses"] == -50.0

    # Net ROI = Gross - Tax - Expenses.
    # Net ROI = 0 - 0 - (-50) = 50.
    assert financial_state["net_roi"] == 50.0
    assert data["new_net_roi"] == 50.0

    # Verify it was added to details
    assert len(financial_state["expenses_detail"]) == 1
    assert financial_state["expenses_detail"][0]["amount"] == -50.0
