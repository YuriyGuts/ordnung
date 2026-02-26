import json
from collections.abc import Sequence
from pathlib import Path

from openai.types.responses import Response
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses import ResponseOutputMessage
from openai.types.responses import ResponseOutputRefusal
from openai.types.responses import ResponseOutputText
from openai.types.responses import ResponseReasoningItem

from ordnung.entities import OrganizeDirectoryResult
from ordnung.entities import OrganizeDirectoryTaskSpec
from ordnung.environment import Environment
from ordnung.llm import LLMClient
from ordnung.tools import Tool
from ordnung.tui import calling_llm_spinner
from ordnung.tui import print_reasoning
from ordnung.tui import print_tool_result


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
        self,
        task_spec: OrganizeDirectoryTaskSpec,
        env: Environment,
    ) -> OrganizeDirectoryResult:
        initial_message = f'Input directory: "{task_spec.dir_path}"'
        self._add_user_message(initial_message)
        while not (agent_result := self.act(env)):
            pass
        return agent_result

    def act(self, env: Environment) -> OrganizeDirectoryResult | None:
        response = self.call_llm()
        self.conversation_context += response.output
        for item in response.output:
            match item:
                case ResponseReasoningItem():
                    self._handle_reasoning(item)
                case ResponseFunctionToolCall():
                    self._handle_tool_call(item, env)
                case ResponseOutputMessage():
                    return self._extract_final_result(item)
                case _:
                    pass

        return None

    def _handle_reasoning(self, item: ResponseReasoningItem) -> None:
        summary_text = "".join(s.text for s in item.summary)
        print_reasoning(summary_text)

    def _handle_tool_call(self, item: ResponseFunctionToolCall, env: Environment) -> None:
        tool_result = env.run_tool(item.name, item.arguments)
        print_tool_result(tool_result)
        function_call_output_item = {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(tool_result),
        }
        self.conversation_context.append(function_call_output_item)

    def _extract_final_result(self, item: ResponseOutputMessage) -> OrganizeDirectoryResult:
        try:
            has_refusals = any(
                cnt for cnt in item.content if isinstance(cnt, ResponseOutputRefusal)
            )
            if has_refusals:
                return OrganizeDirectoryResult(
                    is_success=False,
                    error="The output content has triggered a refusal from the LLM",
                )
            response_text = "".join(
                cnt.text for cnt in item.content if isinstance(cnt, ResponseOutputText)
            )
            agent_response = json.loads(response_text)
            agent_succeeded = agent_response.get("agent_succeeded")
            error_msg = agent_response.get("error")
            return OrganizeDirectoryResult(
                is_success=agent_succeeded,
                error=error_msg,
            )
        except Exception as e:
            return OrganizeDirectoryResult(is_success=False, error=str(e))

    def call_llm(self) -> Response:
        with calling_llm_spinner():
            response = self.llm_client.create_response(
                instructions=self.instructions,
                input=self.conversation_context,
                tools=self.openai_tools,
            )
            return response
