"""Tests for ordnung.agent."""

import json
from unittest.mock import MagicMock

from openai.types.responses import ResponseOutputRefusal
from openai.types.responses import ResponseOutputText

from ordnung.agent import Agent


def test_add_user_message():
    # GIVEN an agent with empty context
    agent = Agent(llm_client=MagicMock(), tools=[])
    assert agent.conversation_context == []

    # WHEN adding a user message
    agent._add_user_message("hello")

    # THEN it appears in the conversation context
    assert agent.conversation_context == [{"role": "user", "content": "hello"}]


def test_add_multiple_user_messages():
    # GIVEN an agent
    agent = Agent(llm_client=MagicMock(), tools=[])

    # WHEN adding multiple messages
    agent._add_user_message("first")
    agent._add_user_message("second")

    # THEN both appear in order
    assert len(agent.conversation_context) == 2
    assert agent.conversation_context[0]["content"] == "first"
    assert agent.conversation_context[1]["content"] == "second"


def test_extract_final_result_success():
    # GIVEN a response message with a successful JSON result
    agent = Agent(llm_client=MagicMock(), tools=[])

    text_content = MagicMock(spec=ResponseOutputText)
    text_content.text = json.dumps({"agent_succeeded": True})
    text_content.__class__ = ResponseOutputText

    msg = MagicMock()
    msg.content = [text_content]

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates success
    assert result.is_success is True
    assert result.error is None


def test_extract_final_result_agent_failure():
    # GIVEN a response message where the agent reports failure
    agent = Agent(llm_client=MagicMock(), tools=[])

    text_content = MagicMock(spec=ResponseOutputText)
    text_content.text = json.dumps({"agent_succeeded": False, "error": "Could not categorize"})
    text_content.__class__ = ResponseOutputText

    msg = MagicMock()
    msg.content = [text_content]

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates failure with the error message
    assert result.is_success is False
    assert result.error == "Could not categorize"


def test_extract_final_result_refusal():
    # GIVEN a response message with a refusal
    agent = Agent(llm_client=MagicMock(), tools=[])

    refusal_content = MagicMock(spec=ResponseOutputRefusal)
    refusal_content.__class__ = ResponseOutputRefusal

    msg = MagicMock()
    msg.content = [refusal_content]

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates failure due to refusal
    assert result.is_success is False
    assert result.error is not None
    assert "refusal" in result.error.lower()


def test_extract_final_result_invalid_json():
    # GIVEN a response message with non-JSON text
    agent = Agent(llm_client=MagicMock(), tools=[])

    text_content = MagicMock(spec=ResponseOutputText)
    text_content.text = "This is not valid JSON"
    text_content.__class__ = ResponseOutputText

    msg = MagicMock()
    msg.content = [text_content]

    # WHEN extracting the result
    result = agent._extract_final_result(msg)

    # THEN it indicates failure
    assert result.is_success is False
    assert result.error is not None


def test_handle_tool_call():
    # GIVEN a tool call item and an environment mock
    agent = Agent(llm_client=MagicMock(), tools=[])

    tool_call_item = MagicMock()
    tool_call_item.name = "ListDirectoryTool"
    tool_call_item.arguments = '{"dir_path": "/tmp"}'
    tool_call_item.call_id = "call_123"

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


def test_system_prompt_loaded():
    # GIVEN a freshly created agent
    agent = Agent(llm_client=MagicMock(), tools=[])

    # THEN the system prompt is loaded and non-empty
    assert agent.system_prompt
    assert len(agent.system_prompt) > 0
