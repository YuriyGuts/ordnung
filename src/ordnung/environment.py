import json
from collections.abc import Sequence

from ordnung.entities import ToolSecurityPolicy
from ordnung.tools import Tool
from ordnung.tui import approval_prompt
from ordnung.tui import print_tool_call_request
from ordnung.tui import user_feedback_prompt


class Environment:
    def __init__(self, tools: Sequence[type[Tool]], sec_policy: ToolSecurityPolicy) -> None:
        self.tool_registry = {tool.get_name(): tool for tool in tools}
        self.sec_policy = sec_policy

    def run_tool(self, name: str, args_raw: str) -> dict:
        if name not in self.tool_registry:
            return {"error": f"Tool {name} does not exist"}

        try:
            parsed_args = json.loads(args_raw)
            tool_cls = self.tool_registry[name]
            tool_obj = tool_cls(**parsed_args)
        except Exception as e:
            return {"error": f"Invalid arguments: {e}"}

        print_tool_call_request(name, parsed_args)
        if name not in self.sec_policy.approved_tool_names:
            approval_result = approval_prompt().lower()
            match approval_result:
                case "y":
                    pass
                case "a":
                    if name not in self.sec_policy.approved_tool_names:
                        self.sec_policy.approved_tool_names.append(name)
                case "f":
                    user_feedback = user_feedback_prompt()
                    return {"error": f"User provided feedback: {user_feedback}"}
                case "n":
                    return {"error": "Tool call rejected by user"}
                case "q":
                    # TODO: handle this better.
                    raise KeyboardInterrupt()

        try:
            tool_output = tool_obj.run(self.sec_policy)
        except Exception as e:
            return {"error": f"Tool call failed: {e}"}

        return tool_output
