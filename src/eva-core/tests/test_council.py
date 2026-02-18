import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from eva_core.services.council import CouncilService, get_council_service

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.ollama_host = "localhost"
    settings.ollama_port = 11434
    settings.council_model_general = "llama3:8b"
    settings.council_model_research = "mistral:7b"
    settings.council_model_banker = "gemma:7b"
    return settings

@pytest.fixture
def council_service(mock_settings):
    with patch("eva_core.services.council.get_settings", return_value=mock_settings):
        # We need to patch httpx.AsyncClient during instantiation as well to avoid real network calls
        # although CouncilService just creates the client, it's safer to patch it.
        # But for simplicity, let's just create it and replace _client immediately.
        # However, to be extra safe, we can patch httpx.AsyncClient itself.
        with patch("httpx.AsyncClient", return_value=AsyncMock()) as MockClient:
            service = CouncilService()
            # The service._client will be the return value of MockClient() which is an AsyncMock
            # But let's make sure we have control over it.
            # The service._client is set to MockClient() return value.
            # Let's just create a fresh AsyncMock for clarity in tests
            service._client = AsyncMock()
            return service

@pytest.mark.asyncio
async def test_prepare_model_same_model(council_service):
    """Test that no API calls are made if the target model is already loaded."""
    council_service.current_model = "llama3:8b"
    result = await council_service.prepare_model("general")
    assert result == "llama3:8b"
    council_service._client.get.assert_not_called()
    council_service._client.post.assert_not_called()

@pytest.mark.asyncio
async def test_prepare_model_not_found_needs_pull(council_service):
    """Test that model is pulled if not found locally."""
    # Setup
    council_service.current_model = "other:model"
    target_model = "llama3:8b" # defined in mock_settings for 'general'

    # Mock tags response (target model not in tags)
    mock_tags_response = MagicMock()
    mock_tags_response.status_code = 200
    mock_tags_response.json.return_value = {"models": [{"name": "existing:model"}]}

    # Mock pull response
    mock_pull_response = MagicMock()
    mock_pull_response.raise_for_status = MagicMock()

    # Mock load response
    mock_load_response = MagicMock()
    mock_load_response.raise_for_status = MagicMock()

    council_service._client.get.return_value = mock_tags_response
    council_service._client.post.side_effect = [mock_pull_response, mock_load_response]

    # Execute
    result = await council_service.prepare_model("general")

    # Verify
    assert result == target_model
    assert council_service.current_model == target_model
    assert council_service.current_role == "general"

    # Verify calls
    council_service._client.get.assert_called_once()
    assert council_service._client.post.call_count == 2

    # Check pull call
    pull_call = council_service._client.post.call_args_list[0]
    assert "api/pull" in pull_call[0][0]
    assert pull_call[1]["json"] == {"name": target_model}

    # Check load call
    load_call = council_service._client.post.call_args_list[1]
    assert "api/generate" in load_call[0][0]
    assert load_call[1]["json"]["model"] == target_model
    assert load_call[1]["json"]["keep_alive"] == "10m"

@pytest.mark.asyncio
async def test_prepare_model_found_locally_skip_pull(council_service):
    """Test that pull is skipped if model is found locally."""
    # Setup
    target_model = "llama3:8b"
    council_service.current_model = "other:model"

    # Mock tags response (target model IS in tags)
    mock_tags_response = MagicMock()
    mock_tags_response.status_code = 200
    mock_tags_response.json.return_value = {"models": [{"name": target_model}]}

    # Mock load response
    mock_load_response = MagicMock()
    mock_load_response.raise_for_status = MagicMock()

    council_service._client.get.return_value = mock_tags_response
    # Only load call, no pull call
    council_service._client.post.return_value = mock_load_response

    # Execute
    result = await council_service.prepare_model("general")

    # Verify
    assert result == target_model
    # Should be called once for loading
    assert council_service._client.post.call_count == 1
    assert "api/generate" in council_service._client.post.call_args[0][0]

@pytest.mark.asyncio
async def test_prepare_model_api_error_fallback(council_service):
    """Test fallback to default model on API error."""
    # Setup
    council_service.current_model = "other:model"

    # Mock tags response failure
    council_service._client.get.side_effect = Exception("API Error")

    # Execute
    result = await council_service.prepare_model("general")

    # Verify fallback to default setting
    assert result == "llama3:8b" # Fallback to general model setting
    # Ensure current model is NOT updated if it failed
    assert council_service.current_model == "other:model"

@pytest.mark.asyncio
async def test_unload_current(council_service):
    """Test unloading the current model."""
    council_service.current_model = "loaded:model"

    # Mock response
    mock_response = MagicMock()
    council_service._client.post.return_value = mock_response

    await council_service.unload_current()

    assert council_service.current_model is None
    council_service._client.post.assert_called_once()
    call_args = council_service._client.post.call_args
    assert "api/generate" in call_args[0][0]
    assert call_args[1]["json"]["model"] == "loaded:model"
    assert call_args[1]["json"]["keep_alive"] == 0

@pytest.mark.asyncio
async def test_unload_current_none(council_service):
    """Test unloading when no model is loaded."""
    council_service.current_model = None
    await council_service.unload_current()
    council_service._client.post.assert_not_called()

@pytest.mark.asyncio
async def test_unload_current_error(council_service):
    """Test error handling during unload."""
    council_service.current_model = "loaded:model"
    council_service._client.post.side_effect = Exception("Unload error")

    await council_service.unload_current()

    # Should handle exception and NOT reset current_model
    assert council_service.current_model == "loaded:model"

def test_get_council_service_singleton():
    """Test that get_council_service returns a singleton."""
    # Reset the global singleton first if it was set by other tests implicitly (though unlikely due to module scope)
    # But better to patch the class itself to verify instantiation
    with patch("eva_core.services.council.CouncilService") as MockService:
        # Also need to reset the global variable in the module
        # Because imports are cached, we need to access the module directly
        import eva_core.services.council
        original_service = eva_core.services.council._council_service
        eva_core.services.council._council_service = None

        try:
            s1 = get_council_service()
            s2 = get_council_service()
            assert s1 is s2
            MockService.assert_called_once()
        finally:
            # Restore
            eva_core.services.council._council_service = original_service
