from typing import Any

from openai import OpenAI
from openai.types.responses import Response


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)

    def create_response(self, instructions: str, input: list[Any], tools: list[Any]) -> Response:
        return self.openai_client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input,
            tools=tools,
        )
