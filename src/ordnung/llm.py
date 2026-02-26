"""Implements LLM API clients."""

from typing import Any

from openai import OpenAI
from openai.types.responses import Response


class LLMClient:
    """A simple client that can talk to an OpenAI-compatible LLM service using the Responses API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        """
        Create a new LLM client.

        Parameters
        ----------
        base_url
            The base URL of the LLM API. Note: you likely need to include the `/v1` suffix.
        api_key
            The API key for LLM API authentication. For Ollama, any value should work.
        model
            The name of the LLM to use.
        """
        self.base_url = base_url
        self.model = model
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)

    def create_response(
        self,
        instructions: str,
        input_items: list[Any],
        tools: list[Any],
    ) -> Response:
        """
        Generate an LLM response for the specified inputs.

        Parameters
        ----------
        instructions
            The top-level instructions for the LLM (system prompt).
        input_items
            The input items (conversation history + tool call results).
        tools
            The tools available to the LLM.

        Returns
        -------
        OpenAI Response object.
        """
        return self.openai_client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
        )
