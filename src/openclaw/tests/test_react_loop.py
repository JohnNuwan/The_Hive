
import sys
from unittest.mock import MagicMock

# Mock torch before any other imports that might use it
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from shared import ChatMessage, MessageRole as Role
from openclaw.core.agent import OpenClawAgent

@pytest.mark.asyncio
async def test_react_loop_basic_flow():
    """Test basic ReAct flow: User -> Agent -> Tool -> Agent -> Final Answer"""

    # Mock LLM Service
    mock_llm = AsyncMock()
    # Sequence of responses:
    # 1. _observe (memory search) -> Mocked inside agent or memory
    # 2. _plan -> "Step 1: check files"
    # 3. ReAct Loop 1 -> "Thought: need to list files.\nAction: fs_list\nAction Input: ."
    # 4. ReAct Loop 2 -> "Thought: found it.\nFinal Answer: The file is there."

    mock_llm.generate_response.side_effect = [
        ("Step 1: Check files.", None),  # Plan
        ("Thought: I need to check files.\nAction: fs_list\nAction Input: .", None), # Tool Call
        ("Final Answer: The file is present.", None) # Final Answer
    ]

    # Mock Memory
    mock_memory = AsyncMock()
    mock_memory.search.return_value = ["Memory 1"]

    with patch("openclaw.core.agent.get_llm_service", return_value=mock_llm), \
         patch("openclaw.core.agent.get_memory_bridge", return_value=mock_memory), \
         patch("openclaw.core.agent.get_skill") as mock_get_skill:

        # Setup Mock Tool
        mock_tool = MagicMock(return_value="file1.txt\nfile2.txt")
        mock_tool._skill_name = "fs_list"
        mock_tool._skill_description = "List files"
        mock_get_skill.return_value = mock_tool

        agent = OpenClawAgent(name="TestAgent", tools=["fs_list"])

        # Run
        result = await agent.run("Check for files")

        # Verification
        assert result == "The file is present."

        # Verify tool was called
        mock_tool.assert_called_once_with(".")

        # Verify history structure
        # 1. User Task
        # 2. Assistant (Tool Call)
        # 3. User (Observation)
        # 4. Assistant (Final Answer)

        # Note: Plan is not added to short_term_history in the code I wrote,
        # but it is used in system prompt.

        # Check if Observation was added
        history_contents = [m.content for m in agent.short_term_history]
        assert any("Observation: file1.txt" in c for c in history_contents)

@pytest.mark.asyncio
async def test_react_loop_no_tool():
    """Test ReAct flow without tool usage"""
    mock_llm = AsyncMock()
    mock_llm.generate_response.side_effect = [
        ("Just answer.", None), # Plan
        ("Final Answer: 42", None) # Immediate Answer
    ]

    mock_memory = AsyncMock()
    mock_memory.search.return_value = []

    with patch("openclaw.core.agent.get_llm_service", return_value=mock_llm), \
         patch("openclaw.core.agent.get_memory_bridge", return_value=mock_memory):

        agent = OpenClawAgent(name="TestAgent", tools=[])
        result = await agent.run("What is 6*7?")

        assert result == "42"
        # Tool should not be called (and no tool mocked)

@pytest.mark.asyncio
async def test_react_loop_max_iterations():
    """Test that loop terminates after max iterations"""
    mock_llm = AsyncMock()
    # Infinite tool calls
    mock_llm.generate_response.return_value = ("Action: dummy\nAction Input: x", None)

    mock_memory = AsyncMock()
    mock_memory.search.return_value = []

    with patch("openclaw.core.agent.get_llm_service", return_value=mock_llm), \
         patch("openclaw.core.agent.get_memory_bridge", return_value=mock_memory), \
         patch("openclaw.core.agent.get_skill") as mock_get_skill:

        mock_tool = MagicMock(return_value="result")
        mock_tool._skill_name = "dummy"
        mock_tool._skill_description = "dummy tool"
        mock_get_skill.return_value = mock_tool

        agent = OpenClawAgent(name="TestAgent", tools=["dummy"])

        result = await agent.run("Infinite loop")

        assert result == "I couldn't complete the task within the limit."
        # Should be called Plan + 5 loops = 6 times
        assert mock_llm.generate_response.call_count >= 5
