"""Tests for ordnung.agent."""

import json
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ordnung.agent import Agent
from ordnung.entities import LLMContentMessage
from ordnung.entities import LLMReasoning
from ordnung.entities import LLMToolCall
from ordnung.entities import OrganizeDirectoryTaskSpec


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with default `make_user_message` behavior."""
    client = MagicMock()
    client.make_user_message.side_effect = lambda text: {"role": "user", "content": text}
    client.make_tool_result.side_effect = lambda call_id, output: {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }
    return client


@pytest.fixture
def agent(mock_llm_client):
    """Create an agent with no tools and a mocked LLM client."""
    return Agent(llm_client=mock_llm_client, tools=[])


def test_add_user_message(agent):
    # GIVEN an agent with empty context
    assert agent.conversation_context == []

    # WHEN adding a user message
    agent._add_user_message("hello")

    # THEN it appears in the conversation context
    assert agent.conversation_context == [{"role": "user", "content": "hello"}]


def test_add_multiple_user_messages(agent):
    # GIVEN an agent

    # WHEN adding multiple messages
    agent._add_user_message("first")
    agent._add_user_message("second")

    # THEN both appear in order
    assert len(agent.conversation_context) == 2
    assert agent.conversation_context[0]["content"] == "first"
    assert agent.conversation_context[1]["content"] == "second"


def test_extract_final_result_success(agent):
    # GIVEN a content message with a successful JSON result
    msg = LLMContentMessage(text=json.dumps({"agent_succeeded": True}), is_refusal=False)

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates success
    assert result.is_success is True
    assert result.error is None


def test_extract_final_result_agent_failure(agent):
    # GIVEN a content message where the agent reports failure
    msg = LLMContentMessage(
        text=json.dumps({"agent_succeeded": False, "error": "Could not categorize"}),
        is_refusal=False,
    )

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates failure with the error message
    assert result.is_success is False
    assert result.error == "Could not categorize"


def test_extract_final_result_refusal(agent):
    # GIVEN a content message with a refusal
    msg = LLMContentMessage(text="", is_refusal=True)

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates failure due to refusal
    assert result.is_success is False
    assert result.error is not None
    assert "refusal" in result.error.lower()


def test_extract_final_result_invalid_json(agent):
    # GIVEN a content message with non-JSON text
    msg = LLMContentMessage(text="This is not valid JSON", is_refusal=False)

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates failure
    assert result.is_success is False
    assert result.error is not None


def test_handle_tool_call(agent, mock_llm_client):
    # GIVEN a tool call item and an environment mock

    tool_call_item = LLMToolCall(
        call_id="call_123",
        name="ListDirectoryTool",
        arguments='{"dir_path": "/tmp"}',
    )

    env = MagicMock()
    env.run_tool.return_value = {"output": []}

    # WHEN handling the tool call
    agent._handle_tool_call(tool_call_item, env)

    # THEN the environment is called with the right arguments
    env.run_tool.assert_called_once_with("ListDirectoryTool", '{"dir_path": "/tmp"}')

    # AND the result is appended to the conversation context
    assert agent.conversation_context == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"output": []}',
        }
    ]


def test_handle_reasoning(agent):
    # GIVEN a reasoning item
    reasoning_item = LLMReasoning(text="raw reasoning")

    # WHEN handling the reasoning item
    with unittest.mock.patch("ordnung.agent.print_reasoning") as mock_print:
        agent._handle_reasoning(reasoning_item)

    # THEN it prints the reasoning text
    mock_print.assert_called_once_with("raw reasoning")


def test_system_prompt_loaded(agent):
    # GIVEN a freshly created agent

    # THEN the system prompt is loaded and non-empty
    assert agent.system_prompt
    assert len(agent.system_prompt) > 0


def test_run_until_done_aborts_after_max_iterations(mock_llm_client, agent):
    # GIVEN an LLM client that never produces a final content message
    client = mock_llm_client
    client.create_response.return_value = MagicMock(items=[], raw_context=[])
    task_spec = OrganizeDirectoryTaskSpec(dir_path=Path("/tmp/test"))
    env = MagicMock()

    # WHEN running the agentic loop with a small iteration limit
    result = agent.run_until_done(task_spec=task_spec, env=env, max_iterations=3)

    # THEN it aborts with a failure result mentioning the iteration limit
    assert result.is_success is False
    assert result.error == "Agent exceeded maximum iterations (3)"
    assert client.create_response.call_count == 3


def test_run_until_done_returns_result_before_max_iterations(mock_llm_client, agent):
    # GIVEN an LLM client that returns a final content message on the second call
    client = mock_llm_client
    success_response = json.dumps({"agent_succeeded": True})
    empty_response = MagicMock(items=[], raw_context=[])
    final_response = MagicMock(
        items=[LLMContentMessage(text=success_response, is_refusal=False)],
        raw_context=[],
    )
    client.create_response.side_effect = [empty_response, final_response]

    task_spec = OrganizeDirectoryTaskSpec(dir_path=Path("/tmp/test"))
    env = MagicMock()

    # WHEN running the agentic loop with a generous iteration limit
    result = agent.run_until_done(task_spec=task_spec, env=env, max_iterations=10)

    # THEN it returns the successful result without exhausting all iterations
    assert result.is_success is True
    assert client.create_response.call_count == 2
