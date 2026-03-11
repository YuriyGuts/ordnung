"""Implements LLM API clients."""

from abc import ABC
from abc import abstractmethod
from typing import Any

from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses import ResponseOutputMessage
from openai.types.responses import ResponseOutputRefusal
from openai.types.responses import ResponseOutputText
from openai.types.responses import ResponseReasoningItem

from ordnung.entities import LLMAPIMode
from ordnung.entities import LLMContentMessage
from ordnung.entities import LLMReasoning
from ordnung.entities import LLMResponse
from ordnung.entities import LLMToolCall


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    def create_response(
        self,
        instructions: str,
        input_items: list[Any],
        tool_schemas: list[dict],
    ) -> LLMResponse:
        """Generate an LLM response for the specified inputs."""

    @abstractmethod
    def make_user_message(self, text: str) -> Any:
        """Create a user message context item."""

    @abstractmethod
    def make_tool_result(self, call_id: str, output: str) -> Any:
        """Create a tool result context item."""


class ResponsesAPILLMClient(LLMClient):
    """LLM client using the OpenAI Responses API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)

    def create_response(
        self,
        instructions: str,
        input_items: list[Any],
        tool_schemas: list[dict],
    ) -> LLMResponse:
        """Generate a normalized LLM response via the Responses API."""
        tools: list[Any] = [self._format_tool(s) for s in tool_schemas]

        response = self.openai_client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
        )

        items = []
        for item in response.output:
            match item:
                case ResponseReasoningItem():
                    items.append(self._parse_reasoning(item))
                case ResponseFunctionToolCall():
                    items.append(self._parse_tool_call(item))
                case ResponseOutputMessage():
                    items.append(self._parse_output_message(item))

        return LLMResponse(items=items, raw_context=list(response.output))

    @staticmethod
    def _format_tool(schema: dict) -> dict:
        """Format a tool schema in the format the Responses API expects."""
        return {"type": "function", **schema}

    @staticmethod
    def _parse_reasoning(item: ResponseReasoningItem) -> LLMReasoning:
        """Extract reasoning text, preferring `content` over `summary`."""
        if item.content:
            text = "".join(c.text for c in item.content or [])
        else:
            text = "".join(s.text for s in item.summary or [])
        return LLMReasoning(text=text)

    @staticmethod
    def _parse_tool_call(item: ResponseFunctionToolCall) -> LLMToolCall:
        """Convert a Responses API function tool call to an `LLMToolCall`."""
        return LLMToolCall(
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    @staticmethod
    def _parse_output_message(item: ResponseOutputMessage) -> LLMContentMessage:
        """Convert a Responses API output message to an `LLMContentMessage`."""
        has_refusal = any(isinstance(cnt, ResponseOutputRefusal) for cnt in item.content)
        if has_refusal:
            return LLMContentMessage(text="", is_refusal=True)
        text = "".join(cnt.text for cnt in item.content if isinstance(cnt, ResponseOutputText))
        return LLMContentMessage(text=text, is_refusal=False)

    def make_user_message(self, text: str) -> Any:
        """Create a Responses API user message context item."""
        return {"role": "user", "content": text}

    def make_tool_result(self, call_id: str, output: str) -> Any:
        """Create a Responses API tool result context item."""
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }


class CompletionsAPILLMClient(LLMClient):
    """LLM client using the OpenAI Chat Completions API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)

    def create_response(
        self,
        instructions: str,
        input_items: list[Any],
        tool_schemas: list[dict],
    ) -> LLMResponse:
        """Generate a normalized LLM response via the Chat Completions API."""
        # Prepend system message and pass conversation context as messages.
        messages = [{"role": "system", "content": instructions}]
        messages.extend(input_items)

        # Append tool definitions.
        tools = [self._format_tool(s) for s in tool_schemas]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        completion = self.openai_client.chat.completions.create(**kwargs)
        message = completion.choices[0].message

        return LLMResponse(
            items=self._parse_message(message),
            raw_context=[self._build_raw_context(message)],
        )

    @staticmethod
    def _parse_message(message: Any) -> list[LLMReasoning | LLMToolCall | LLMContentMessage]:
        """Extract normalized response items from a Chat Completions message."""
        items: list[LLMReasoning | LLMToolCall | LLMContentMessage] = []

        # Extract reasoning content if present (some providers support it).
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            items.append(LLMReasoning(text=reasoning_content))

        # Process tool calls if present.
        if message.tool_calls:
            items.extend(
                LLMToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in message.tool_calls
            )

        # Process refusal if present.
        if message.refusal:
            items.append(LLMContentMessage(text=message.refusal, is_refusal=True))
        elif message.content:
            items.append(LLMContentMessage(text=message.content, is_refusal=False))

        return items

    @staticmethod
    def _build_raw_context(message: Any) -> dict[str, Any]:
        """Serialize a Chat Completions message into a dict for re-injection as context."""
        assistant_dict: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_dict["content"] = message.content
        if message.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return assistant_dict

    @staticmethod
    def _format_tool(schema: dict) -> dict:
        """Format a tool schema in the format the Completions API expects."""
        return {"type": "function", "function": schema}

    def make_user_message(self, text: str) -> Any:
        """Create a Chat Completions API user message context item."""
        return {"role": "user", "content": text}

    def make_tool_result(self, call_id: str, output: str) -> Any:
        """Create a Chat Completions API tool result context item."""
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output,
        }


def create_llm_client(
    api_mode: LLMAPIMode,
    base_url: str,
    api_key: str,
    model: str,
) -> LLMClient:
    """
    Create an LLM client for the specified API mode.

    Parameters
    ----------
    api_mode
        The LLM API mode to use.
    base_url
        The base URL of the LLM API.
    api_key
        The API key for LLM API authentication.
    model
        The name of the LLM to use.
    """
    match api_mode:
        case LLMAPIMode.COMPLETIONS:
            return CompletionsAPILLMClient(base_url=base_url, api_key=api_key, model=model)
        case LLMAPIMode.RESPONSES:
            return ResponsesAPILLMClient(base_url=base_url, api_key=api_key, model=model)
        case _:
            raise ValueError(f"Unknown LLM API mode: {api_mode}")
