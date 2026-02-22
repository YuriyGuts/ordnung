import json
from collections.abc import Sequence
from pathlib import Path

from openai.types.responses import Response

from ordnung.entities import OrganizeDirectoryResult
from ordnung.entities import OrganizeDirectoryTaskSpec
from ordnung.environment import Environment
from ordnung.llm import LLMClient
from ordnung.tools import Tool


class Agent:
    def __init__(self, llm_client: LLMClient, tools: Sequence[type[Tool]]) -> None:
        self.llm_client = llm_client
        self.instructions = self._get_system_prompt()
        self.openai_tools = [self._create_openai_tool(tool) for tool in tools]
        self.conversation_context = []

    def _create_openai_tool(self, tool: type[Tool]) -> dict:
        return {
            "type": "function",
            "name": tool.get_name(),
            "description": tool.get_description(),
            "parameters": tool.model_json_schema(),
        }

    def _get_system_prompt(self) -> str:
        current_file_path = Path(__file__)
        return (current_file_path.parent / "system_prompt.md").read_text(encoding="utf-8")

    def _add_user_message(self, msg: str) -> None:
        self.conversation_context.append(
            {"role": "user", "content": msg},
        )

    def run_until_done(
        self, task_spec: OrganizeDirectoryTaskSpec, env: Environment
    ) -> OrganizeDirectoryResult:
        initial_message = f"Input directory: {task_spec.dir_path}"
        self._add_user_message(initial_message)
        while not (agent_result := self.run_iteration(env)):
            pass
        return agent_result

    def run_iteration(self, env: Environment) -> OrganizeDirectoryResult | None:
        response = self.act()
        self.conversation_context += response.output
        for item in response.output:
            match item.type:
                case "reasoning":
                    print("[Thinking]")
                    print(item.summary[0].text)
                case "message":
                    try:
                        # TODO: harden this (expect unhappy paths).
                        agent_response = json.loads(item.content[0].text)
                        agent_succeeded = agent_response.get("agent_succeeded")
                        error_msg = agent_response.get("error")
                        return OrganizeDirectoryResult(
                            is_success=agent_succeeded,
                            error=error_msg,
                        )
                    except Exception as e:
                        return OrganizeDirectoryResult(is_success=False, error=str(e))
                # TODO: parse tool calls.
                case _:
                    pass

        return None
        return OrganizeDirectoryResult(is_success=True)

    def act(self) -> Response:
        print("Calling the LLM...")
        response = self.llm_client.create_response(
            instructions=self.instructions,
            input=self.conversation_context,
            tools=self.openai_tools,
        )
        return response
