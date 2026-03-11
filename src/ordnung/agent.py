"""Implements the main behavior of the agent."""

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
from ordnung.tui import print_content_response
from ordnung.tui import print_reasoning
from ordnung.tui import print_tool_result


class Agent:
    """
    An LLM-powered agent.

    Runs the agentic loop, delegating the decision-making logic to the LLM.
    When the LLM returns tool call requests, asks the environment to execute the tool.
    """

    def __init__(self, llm_client: LLMClient, tools: Sequence[type[Tool]]) -> None:
        """
        Create a new agent.

        Parameters
        ----------
        llm_client
            The LLM client to use for LLM calls.
        tools
            The tool classes available to the agent.
        """
        self.llm_client = llm_client
        self.system_prompt = self._get_system_prompt()
        self.openai_tools = [tool.to_openai_format() for tool in tools]
        self.conversation_context = []

    def _get_system_prompt(self) -> str:
        """Read the system prompt from a file."""
        current_file_path = Path(__file__)
        return (current_file_path.parent / "system_prompt.md").read_text(encoding="utf-8")

    def _add_user_message(self, msg: str) -> None:
        """Add a user message to the conversation context."""
        self.conversation_context.append(
            {"role": "user", "content": msg},
        )

    def run_until_done(
        self,
        task_spec: OrganizeDirectoryTaskSpec,
        env: Environment,
    ) -> OrganizeDirectoryResult:
        """
        Run the agentic loop until it reaches a final result (success or failure).

        Parameters
        ----------
        task_spec
            The specification of the task to perform.
        env
            The environment to execute the tools in.

        Returns
        -------
        The result of performing the task.
        """
        initial_message = f'Input directory: "{task_spec.dir_path}"'
        self._add_user_message(initial_message)
        while not (final_result := self._act(env)):
            pass
        return final_result

    def _act(self, env: Environment) -> OrganizeDirectoryResult | None:
        """
        Run a single iteration of the agentic loop.

        Use the LLM to decide the next action based on the conversation context accumulated so far.
        Handle different types of LLM outputs (reasoning, tool call, final content output).

        Parameters
        ----------
        env
            The environment to execute the tools in.

        Returns
        -------
        Final task result if a final state has been reached after this iteration.
        Otherwise, return None.
        """
        # Run LLM inference and receive an OpenAI Responses API object.
        response = self._call_llm()
        # Append LLM outputs to the conversation context so that they are passed on the next turn.
        self.conversation_context += response.output

        # Handle each output item type individually.
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

    def _call_llm(self) -> Response:
        """Send the current context to the LLM API for inference."""
        with calling_llm_spinner():
            response = self.llm_client.create_response(
                instructions=self.system_prompt,
                input_items=self.conversation_context,
                tools=self.openai_tools,
            )
            return response

    def _handle_reasoning(self, item: ResponseReasoningItem) -> None:
        """Print the reasoning output from the LLM."""
        # Prefer `content` (raw reasoning text) over `summary`.
        if item.content:
            reasoning_text = "".join(c.text for c in item.content)
        else:
            reasoning_text = "".join(s.text for s in item.summary)
        print_reasoning(reasoning_text)

    def _handle_tool_call(self, item: ResponseFunctionToolCall, env: Environment) -> None:
        """Execute a tool call and append the result to the conversation context."""
        tool_result = env.run_tool(item.name, item.arguments)
        print_tool_result(tool_result)

        # Append the tool call result to the context, linking it by `call_id`.
        function_call_output_item = {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(tool_result),
        }
        self.conversation_context.append(function_call_output_item)

    def _extract_final_result(self, item: ResponseOutputMessage) -> OrganizeDirectoryResult:
        """Extract the overall task result from the LLM's content response."""
        try:
            # Check for refusals (triggered LLM safety guardrails) and terminate if we found any.
            has_refusals = any(isinstance(cnt, ResponseOutputRefusal) for cnt in item.content)
            if has_refusals:
                return OrganizeDirectoryResult(
                    is_success=False,
                    error="The prompt has triggered a refusal from the LLM",
                )

            # Otherwise, process the regular text responses.
            response_text = "".join(
                cnt.text for cnt in item.content if isinstance(cnt, ResponseOutputText)
            )
            print_content_response(response_text)

            # The LLM may not follow the instructions closely and produce an invalid final response.
            # Therefore, we are parsing the content defensively here.
            agent_response = json.loads(response_text)
            agent_succeeded = agent_response.get("agent_succeeded")
            error_msg = agent_response.get("error")

            return OrganizeDirectoryResult(
                is_success=agent_succeeded,
                error=error_msg,
            )
        except Exception as e:
            return OrganizeDirectoryResult(is_success=False, error=str(e))
