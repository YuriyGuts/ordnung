"""Implements the tool execution environment."""

import json
from collections.abc import Sequence

from ordnung.entities import ToolSecurityPolicy
from ordnung.tools import Tool
from ordnung.tui import approval_prompt
from ordnung.tui import print_tool_call_request
from ordnung.tui import user_feedback_prompt


class Environment:
    """
    An environment that can execute tool call requests.

    The idea is that the Environment acts like an "OS kernel" that executes "privileged code"
    (tool implementations) when requested by the Agent.
    """

    def __init__(self, tools: Sequence[type[Tool]], sec_policy: ToolSecurityPolicy) -> None:
        """
        Create a new environment.

        Parameters
        ----------
        tools
            The tool classes this environment can execute.
        sec_policy
            The initial security policy configuration for the environment.
        """
        self.tool_registry = {tool.get_name(): tool for tool in tools}
        self.sec_policy = sec_policy

    def run_tool(self, name: str, args_raw: str) -> dict:
        """
        Execute a tool and capture its results.

        Parameters
        ----------
        name
            The name of the tool to execute.
        args_raw
            The raw arguments for the tool call, as specified by the LLM.

        Returns
        -------
        A dict containing `error` in case of a validation/runtime error.
        Otherwise, the output dict returned by the tool implementation.

        Raises
        ------
        KeyboardInterrupt
            If the user has requested to quit the session.
        """
        # The LLM may hallucinate and request invalid tool names or arguments.
        # We parse the inputs very carefully.
        if name not in self.tool_registry:
            return {"error": f"Tool {name} does not exist"}

        try:
            parsed_args = json.loads(args_raw)
            tool_cls = self.tool_registry[name]
            tool_obj = tool_cls(**parsed_args)
        except Exception as e:
            return {"error": f"Invalid arguments: {e}"}

        print_tool_call_request(name, parsed_args)

        # Check if the security policy requires user approval of the tool call.
        # If yes, present the approval prompt.
        if name not in self.sec_policy.approved_tool_names:
            approval_result = approval_prompt().lower()
            match approval_result:
                case "y":
                    # User approved the tool call: do nothing (let the tool run).
                    pass
                case "a":
                    # User approved this and all future calls of this tool:
                    # add the tool name to the allow-list and let the tool run.
                    if name not in self.sec_policy.approved_tool_names:
                        self.sec_policy.approved_tool_names.append(name)
                case "f":
                    # User provided steering/guidance feedback: reject the call
                    # and return the feedback as a tool call result.
                    user_feedback = user_feedback_prompt()
                    return {"error": f"User provided feedback: {user_feedback}"}
                case "n":
                    # User rejected the tool call without providing any feedback.
                    return {"error": "Tool call rejected by user"}
                case "q":
                    # User decided to quit the session.
                    raise KeyboardInterrupt()

        try:
            tool_output = tool_obj.run(self.sec_policy)
        except Exception as e:
            return {"error": f"Tool call failed: {e}"}

        return tool_output
