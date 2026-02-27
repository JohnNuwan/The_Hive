import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from uuid import uuid4

# Import necessary types
from shared.models import ChatMessage, MessageRole

# Mock dependencies before importing the service
with patch("eva_core.services.council.get_council_service"), \
     patch("shared.memory_bridge.get_memory_bridge"):
    from eva_core.services.llm import LLMService, get_llm_service

@pytest.fixture
def mock_settings():
    with patch("eva_core.services.llm.get_settings") as mock_settings_fn:
        settings = MagicMock()
        settings.llm_backend = "ollama"
        settings.ollama_host = "localhost"
        settings.ollama_port = 11434
        settings.ollama_model = "llama3.2:1b"
        settings.vllm_host = "vllm_host"
        settings.vllm_port = 8000
        settings.vllm_model = "vllm_model"
        settings.council_model_general = "general_model"

        mock_settings_fn.return_value = settings
        yield settings

@pytest.fixture
def llm_service(mock_settings):
    # Mock CouncilService and MemoryBridge
    with patch("eva_core.services.llm.get_council_service") as mock_council_fn, \
         patch("eva_core.services.llm.get_memory_bridge") as mock_memory_fn:

        mock_council = AsyncMock()
        mock_council.prepare_model.return_value = "prepared_model"
        mock_council_fn.return_value = mock_council

        mock_memory = AsyncMock()
        mock_memory.search.return_value = []
        mock_memory_fn.return_value = mock_memory

        service = LLMService(
            host="test_host",
            port=1234,
            model="test_model",
            use_ollama=True
        )

        # Mock httpx client
        service._client = AsyncMock(spec=httpx.AsyncClient)

        return service

@pytest.mark.asyncio
async def test_initialization(mock_settings):
    """Test LLMService initialization with settings."""
    # Ensure cache is cleared before starting
    get_llm_service.cache_clear()

    # Test Ollama config
    mock_settings.llm_backend = "ollama"
    service = get_llm_service()
    assert service.host == "localhost"
    assert service.port == 11434
    assert service.model == "llama3.2:1b"
    assert service.use_ollama is True

    # Reset lru_cache for get_llm_service to test vLLM
    get_llm_service.cache_clear()

    # Test vLLM config
    mock_settings.llm_backend = "vllm"
    service_vllm = get_llm_service()
    assert service_vllm.host == "vllm_host"
    assert service_vllm.port == 8000
    assert service_vllm.model == "vllm_model"
    assert service_vllm.use_ollama is False

@pytest.mark.asyncio
async def test_generate_response_ollama(llm_service):
    """Test happy path with Ollama."""
    llm_service.use_ollama = True
    llm_service.council.prepare_model.return_value = "council_model"
    llm_service.memory.search.return_value = ["Memory 1", "Memory 2"]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Ollama response"}
    llm_service._client.post.return_value = mock_response

    messages = [
        ChatMessage(session_id=uuid4(), role=MessageRole.USER, content="Hello")
    ]

    response, thought = await llm_service.generate_response(messages, role="general")

    assert response == "Ollama response"
    assert thought is None

    # Verify council call
    llm_service.council.prepare_model.assert_awaited_with("general")

    # Verify memory search
    llm_service.memory.search.assert_awaited_with("Hello", limit=3)

    # Verify API call
    llm_service._client.post.assert_called_once()
    args, kwargs = llm_service._client.post.call_args
    assert kwargs["json"]["model"] == "council_model"
    assert "Memory 1" in kwargs["json"]["prompt"]
    assert "Memory 2" in kwargs["json"]["prompt"]

@pytest.mark.asyncio
async def test_generate_response_vllm(llm_service):
    """Test happy path with vLLM."""
    llm_service.use_ollama = False
    llm_service.council.prepare_model.return_value = "council_model"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "vLLM response"}}]
    }
    llm_service._client.post.return_value = mock_response

    messages = [
        ChatMessage(session_id=uuid4(), role=MessageRole.USER, content="Hello")
    ]

    response, thought = await llm_service.generate_response(messages)

    assert response == "vLLM response"

    # Verify API call
    llm_service._client.post.assert_called_once()
    args, kwargs = llm_service._client.post.call_args
    assert kwargs["json"]["model"] == "council_model"
    # Verify format for vLLM (OpenAI compatible)
    assert "messages" in kwargs["json"]
    assert kwargs["json"]["messages"][0]["role"] == "user"
    assert kwargs["json"]["messages"][0]["content"] == "Hello"

@pytest.mark.asyncio
async def test_generate_response_thought_extraction(llm_service):
    """Test <thought> tag extraction."""
    llm_service.use_ollama = True

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "<thought>Thinking process...</thought> Final answer"
    }
    llm_service._client.post.return_value = mock_response

    messages = [ChatMessage(session_id=uuid4(), role=MessageRole.USER, content="Hi")]

    response, thought = await llm_service.generate_response(messages)

    assert response == "Final answer"
    assert thought == "Thinking process..."

@pytest.mark.asyncio
async def test_generate_response_memory_failure(llm_service):
    """Test resilience when memory retrieval fails."""
    llm_service.memory.search.side_effect = Exception("Memory Error")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Response despite memory error"}
    llm_service._client.post.return_value = mock_response

    messages = [ChatMessage(session_id=uuid4(), role=MessageRole.USER, content="Hi")]

    # Should not raise exception
    response, _ = await llm_service.generate_response(messages)

    assert response == "Response despite memory error"

@pytest.mark.asyncio
async def test_generate_response_connection_error(llm_service):
    """Test handling of httpx.ConnectError."""
    llm_service._client.post.side_effect = httpx.ConnectError("Connection refused")

    messages = [ChatMessage(session_id=uuid4(), role=MessageRole.USER, content="Hi")]

    response, thought = await llm_service.generate_response(messages)

    assert "[Mode Dev]" in response
    assert thought is None

@pytest.mark.asyncio
async def test_generate_response_general_exception(llm_service):
    """Test handling of generic exceptions."""
    llm_service._client.post.side_effect = Exception("Unexpected error")

    messages = [ChatMessage(session_id=uuid4(), role=MessageRole.USER, content="Hi")]

    response, thought = await llm_service.generate_response(messages)

    assert "Désolé, j'ai rencontré une erreur" in response
    assert "Unexpected error" in response
    assert thought is None
