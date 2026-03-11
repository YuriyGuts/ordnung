"""Implements the main behavior of the agent."""

import json
from collections.abc import Sequence
from pathlib import Path

from ordnung.entities import DEFAULT_MAX_ITERATIONS
from ordnung.entities import LLMContentMessage
from ordnung.entities import LLMReasoning
from ordnung.entities import LLMResponse
from ordnung.entities import LLMToolCall
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
        self.tool_schemas = [tool.to_schema() for tool in tools]
        self.conversation_context = []

    def _get_system_prompt(self) -> str:
        """Read the system prompt from a file."""
        current_file_path = Path(__file__)
        return (current_file_path.parent / "system_prompt.md").read_text(encoding="utf-8")

    def _add_user_message(self, msg: str) -> None:
        """Add a user message to the conversation context."""
        self.conversation_context.append(
            self.llm_client.make_user_message(msg),
        )

    def run_until_done(
        self,
        task_spec: OrganizeDirectoryTaskSpec,
        env: Environment,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> OrganizeDirectoryResult:
        """
        Run the agentic loop until it reaches a final result (success or failure).

        Parameters
        ----------
        task_spec
            The specification of the task to perform.
        env
            The environment to execute the tools in.
        max_iterations
            The maximum allowed number of agentic loop iterations before aborting.

        Returns
        -------
        The result of performing the task.
        """
        initial_message = f'Input directory: "{task_spec.dir_path}"'
        self._add_user_message(initial_message)

        for _ in range(max_iterations):
            if final_result := self._act(env):
                return final_result

        return OrganizeDirectoryResult(
            is_success=False,
            error=f"Agent exceeded maximum iterations ({max_iterations})",
        )

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
        # Run LLM inference and receive an API-agnostic response.
        response = self._call_llm()

        # Append the raw LLM output items to the context so they are passed on the next turn.
        self.conversation_context += response.raw_context

        # Check if the response contains any tool calls.
        # If it does, content messages are informational (e.g., "Now I have a good understanding")
        # rather than a final result.
        has_tool_calls = any(isinstance(item, LLMToolCall) for item in response.items)

        # Handle each output item type individually.
        for item in response.items:
            match item:
                case LLMReasoning():
                    self._handle_reasoning(item)
                case LLMToolCall():
                    self._handle_tool_call(item, env)
                case LLMContentMessage() if not has_tool_calls:
                    return self._extract_final_result(item)
                case LLMContentMessage():
                    print_content_response(item.text)

        return None

    def _call_llm(self) -> LLMResponse:
        """Send the current context to the LLM API for inference."""
        with calling_llm_spinner():
            response = self.llm_client.create_response(
                instructions=self.system_prompt,
                input_items=self.conversation_context,
                tool_schemas=self.tool_schemas,
            )
            return response

    def _handle_reasoning(self, item: LLMReasoning) -> None:
        """Print the reasoning output from the LLM."""
        print_reasoning(item.text)

    def _handle_tool_call(self, item: LLMToolCall, env: Environment) -> None:
        """Execute a tool call and append the result to the conversation context."""
        tool_result = env.run_tool(item.name, item.arguments)
        print_tool_result(tool_result)

        # Append the tool call result to the context, linking it by `call_id`.
        tool_result_item = self.llm_client.make_tool_result(
            call_id=item.call_id,
            output=json.dumps(tool_result),
        )
        self.conversation_context.append(tool_result_item)

    def _extract_final_result(self, item: LLMContentMessage) -> OrganizeDirectoryResult:
        """Extract the overall task result from the LLM's content response."""
        try:
            if item.is_refusal:
                return OrganizeDirectoryResult(
                    is_success=False,
                    error="The prompt has triggered a refusal from the LLM",
                )

            print_content_response(item.text)

            # The LLM may not follow the instructions closely and produce an invalid final response.
            # Therefore, we are parsing the content defensively here.
            agent_response = json.loads(item.text)
            agent_succeeded = agent_response.get("agent_succeeded")
            error_msg = agent_response.get("error")

            return OrganizeDirectoryResult(
                is_success=agent_succeeded,
                error=error_msg,
            )
        except Exception as e:
            return OrganizeDirectoryResult(is_success=False, error=str(e))
