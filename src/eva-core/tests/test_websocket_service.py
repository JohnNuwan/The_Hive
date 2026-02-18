
import sys
from unittest.mock import MagicMock, AsyncMock

# Mock main before importing websocket service
if "eva_core.main" not in sys.modules:
    sys.modules["eva_core.main"] = MagicMock()

import pytest
from eva_core.services.websocket import WebSocketService

@pytest.fixture
def websocket_service():
    return WebSocketService()

@pytest.fixture
def mock_websocket():
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    return ws

@pytest.mark.asyncio
async def test_connect(websocket_service, mock_websocket):
    """Test accepting a new WebSocket connection."""
    await websocket_service.connect(mock_websocket)

    mock_websocket.accept.assert_awaited_once()
    assert mock_websocket in websocket_service.active_connections
    assert websocket_service.connection_count == 1

@pytest.mark.asyncio
async def test_disconnect(websocket_service, mock_websocket):
    """Test disconnecting a WebSocket."""
    await websocket_service.connect(mock_websocket)
    await websocket_service.disconnect(mock_websocket)

    assert mock_websocket not in websocket_service.active_connections
    assert websocket_service.connection_count == 0

@pytest.mark.asyncio
async def test_broadcast(websocket_service):
    """Test broadcasting a message to all connected clients."""
    ws1 = AsyncMock()
    ws2 = AsyncMock()

    await websocket_service.connect(ws1)
    await websocket_service.connect(ws2)

    message = {"type": "test", "data": "hello"}
    await websocket_service.broadcast(message)

    # Verify message sent to both
    expected_payload = '{"type": "test", "data": "hello"}'
    ws1.send_text.assert_awaited_once_with(expected_payload)
    ws2.send_text.assert_awaited_once_with(expected_payload)

@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections(websocket_service):
    """Test that failed connections are removed during broadcast."""
    ws1 = AsyncMock()
    ws2 = AsyncMock()

    # ws2 will fail
    ws2.send_text.side_effect = Exception("Connection closed")

    await websocket_service.connect(ws1)
    await websocket_service.connect(ws2)

    message = {"data": "test"}
    await websocket_service.broadcast(message)

    # ws1 should receive message
    ws1.send_text.assert_awaited_once()

    # ws2 should be removed
    assert ws1 in websocket_service.active_connections
    assert ws2 not in websocket_service.active_connections
    assert websocket_service.connection_count == 1

@pytest.mark.asyncio
async def test_broadcast_event(websocket_service):
    """Test broadcasting a structured event."""
    ws = AsyncMock()
    await websocket_service.connect(ws)

    await websocket_service.broadcast_event("ALERT", {"level": "critical"})

    expected_payload = '{"type": "ALERT", "payload": {"level": "critical"}}'
    ws.send_text.assert_awaited_once_with(expected_payload)

@pytest.mark.asyncio
async def test_send_personal(websocket_service, mock_websocket):
    """Test sending a message to a specific client."""
    await websocket_service.connect(mock_websocket)

    message = {"private": "message"}
    await websocket_service.send_personal(mock_websocket, message)

    mock_websocket.send_text.assert_awaited_once_with('{"private": "message"}')

@pytest.mark.asyncio
async def test_send_personal_error(websocket_service, mock_websocket):
    """Test error handling when sending personal message."""
    await websocket_service.connect(mock_websocket)

    mock_websocket.send_text.side_effect = Exception("Failed")

    message = {"data": "test"}
    await websocket_service.send_personal(mock_websocket, message)

    # Should be removed from connections
    assert mock_websocket not in websocket_service.active_connections
