"""Tests for ordnung.llm."""

from unittest.mock import MagicMock
from unittest.mock import patch

from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses import ResponseOutputMessage
from openai.types.responses import ResponseOutputRefusal
from openai.types.responses import ResponseOutputText
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Content as ReasoningContent

from ordnung.entities import LLMAPIMode
from ordnung.entities import LLMContentMessage
from ordnung.entities import LLMReasoning
from ordnung.entities import LLMToolCall
from ordnung.llm import CompletionsAPILLMClient
from ordnung.llm import ResponsesAPILLMClient
from ordnung.llm import create_llm_client


def _make_responses_client(output_items):
    """Create a `ResponsesAPILLMClient` with a mocked OpenAI client returning the given items."""
    with patch("ordnung.llm.OpenAI") as mock_openai_cls:
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_response = MagicMock()
        mock_response.output = output_items
        mock_openai.responses.create.return_value = mock_response

        client = ResponsesAPILLMClient(base_url="http://test", api_key="key", model="m")
    return client


def _make_completions_client(
    *,
    tool_calls=None,
    content=None,
    refusal=None,
    reasoning_content=None,
):
    """Create a `CompletionsAPILLMClient` with a mocked OpenAI client."""
    with patch("ordnung.llm.OpenAI") as mock_openai_cls:
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        message = MagicMock()
        message.tool_calls = tool_calls
        message.content = content
        message.refusal = refusal
        message.reasoning_content = reasoning_content

        choice = MagicMock()
        choice.message = message
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[choice])

        client = CompletionsAPILLMClient(base_url="http://test", api_key="key", model="m")
    return client


class TestResponsesAPILLMClient:
    def test_translates_tool_call(self):
        # GIVEN a Responses API client and a response with a tool call
        tool_call = ResponseFunctionToolCall(
            id="x",
            call_id="call_1",
            name="ListDirectoryTool",
            arguments='{"dir_path": "/tmp"}',
            type="function_call",
        )
        client = _make_responses_client([tool_call])

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN it produces a normalized LLMToolCall
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, LLMToolCall)
        assert item.call_id == "call_1"
        assert item.name == "ListDirectoryTool"
        assert item.arguments == '{"dir_path": "/tmp"}'

    def test_translates_content_message(self):
        # GIVEN a Responses API client and a response with a text content message
        text_content = ResponseOutputText(
            text='{"agent_succeeded": true}',
            type="output_text",
            annotations=[],
        )
        msg = ResponseOutputMessage(
            id="x",
            content=[text_content],
            role="assistant",
            status="completed",
            type="message",
        )
        client = _make_responses_client([msg])

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN it produces a normalized LLMContentMessage
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, LLMContentMessage)
        assert item.text == '{"agent_succeeded": true}'
        assert item.is_refusal is False

    def test_translates_refusal(self):
        # GIVEN a Responses API client and a response with a refusal
        refusal = ResponseOutputRefusal(refusal="no", type="refusal")
        msg = ResponseOutputMessage(
            id="x",
            content=[refusal],
            role="assistant",
            status="completed",
            type="message",
        )
        client = _make_responses_client([msg])

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN it produces a normalized LLMContentMessage with refusal
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, LLMContentMessage)
        assert item.is_refusal is True

    def test_translates_reasoning(self):
        # GIVEN a Responses API client and a response with a reasoning item
        reasoning = ResponseReasoningItem(
            id="x",
            content=[ReasoningContent(text="thinking...", type="reasoning_text")],
            summary=[],
            type="reasoning",
        )
        client = _make_responses_client([reasoning])

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN it produces a normalized LLMReasoning
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, LLMReasoning)
        assert item.text == "thinking..."

    def test_make_user_message(self):
        client = _make_responses_client([])
        assert client.make_user_message("hello") == {"role": "user", "content": "hello"}

    def test_make_tool_result(self):
        client = _make_responses_client([])
        result = client.make_tool_result("call_1", '{"output": []}')
        assert result == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"output": []}',
        }


class TestCompletionsAPILLMClient:
    def test_translates_tool_call(self):
        # GIVEN a Completions API client and a response with tool calls
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "ListDirectoryTool"
        tc.function.arguments = '{"dir_path": "/tmp"}'
        client = _make_completions_client(tool_calls=[tc])

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN it produces a normalized LLMToolCall
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, LLMToolCall)
        assert item.call_id == "call_1"
        assert item.name == "ListDirectoryTool"

    def test_translates_content(self):
        # GIVEN a Completions API client and a response with text content
        client = _make_completions_client(content='{"agent_succeeded": true}')

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN it produces a normalized LLMContentMessage
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, LLMContentMessage)
        assert item.text == '{"agent_succeeded": true}'
        assert item.is_refusal is False

    def test_translates_refusal(self):
        # GIVEN a Completions API client and a response with a refusal
        client = _make_completions_client(refusal="I cannot do that")

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN it produces a normalized LLMContentMessage with refusal
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, LLMContentMessage)
        assert item.is_refusal is True

    def test_wraps_tools(self):
        # GIVEN a Completions API client
        client = _make_completions_client(content="done")

        # WHEN creating a response with tools
        tool_def = {"name": "foo", "description": "bar", "parameters": {}}
        client.create_response(instructions="test", input_items=[], tool_schemas=[tool_def])

        # THEN the tools are wrapped in the Completions API envelope
        call_kwargs = client.openai_client.chat.completions.create.call_args
        expected_tools = [{"type": "function", "function": tool_def}]
        assert call_kwargs.kwargs["tools"] == expected_tools

    def test_prepends_system_message(self):
        # GIVEN a Completions API client
        client = _make_completions_client(content="done")

        # WHEN creating a response with instructions and a user message
        user_msg = {"role": "user", "content": "hello"}
        client.create_response(
            instructions="system prompt", input_items=[user_msg], tool_schemas=[]
        )

        # THEN the system message is prepended to the messages
        call_kwargs = client.openai_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "system prompt"}
        assert messages[1] == user_msg

    def test_make_user_message(self):
        client = _make_completions_client()
        assert client.make_user_message("hello") == {"role": "user", "content": "hello"}

    def test_translates_reasoning(self):
        # GIVEN a Completions API client and a response with reasoning content
        client = _make_completions_client(
            reasoning_content="Let me think about this...", content='{"agent_succeeded": true}'
        )

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN reasoning appears as the first item, followed by the content message
        assert len(result.items) == 2
        assert isinstance(result.items[0], LLMReasoning)
        assert result.items[0].text == "Let me think about this..."
        assert isinstance(result.items[1], LLMContentMessage)

    def test_reasoning_with_tool_calls(self):
        # GIVEN a Completions API client and a response with reasoning and tool calls
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "ListDirectoryTool"
        tc.function.arguments = '{"dir_path": "/tmp"}'
        client = _make_completions_client(
            reasoning_content="I should list the directory.", tool_calls=[tc]
        )

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN reasoning appears before the tool call
        assert len(result.items) == 2
        assert isinstance(result.items[0], LLMReasoning)
        assert result.items[0].text == "I should list the directory."
        assert isinstance(result.items[1], LLMToolCall)

    def test_content_preserved_alongside_tool_calls(self):
        # GIVEN a Completions API client and a response with both content and tool calls
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "ListDirectoryTool"
        tc.function.arguments = '{"dir_path": "/tmp"}'
        client = _make_completions_client(
            content="Let me list the directory first.",
            tool_calls=[tc],
        )

        # WHEN creating a response
        result = client.create_response(instructions="test", input_items=[], tool_schemas=[])

        # THEN both the tool call and the content message are present
        assert len(result.items) == 2
        assert isinstance(result.items[0], LLMToolCall)
        assert isinstance(result.items[1], LLMContentMessage)
        assert result.items[1].text == "Let me list the directory first."
        assert result.items[1].is_refusal is False

    def test_make_tool_result(self):
        client = _make_completions_client()
        result = client.make_tool_result("call_1", '{"output": []}')
        assert result == {"role": "tool", "tool_call_id": "call_1", "content": '{"output": []}'}


class TestCreateLLMClient:
    @patch("ordnung.llm.OpenAI")
    def test_responses(self, mock_openai_cls):
        client = create_llm_client(LLMAPIMode.RESPONSES, "http://test", "key", "m")
        assert isinstance(client, ResponsesAPILLMClient)

    @patch("ordnung.llm.OpenAI")
    def test_completions(self, mock_openai_cls):
        client = create_llm_client(LLMAPIMode.COMPLETIONS, "http://test", "key", "m")
        assert isinstance(client, CompletionsAPILLMClient)
