"""Implements the top-level operations for directory organization."""

from pathlib import Path

from ordnung.agent import Agent
from ordnung.entities import OrganizeDirectoryResult
from ordnung.entities import OrganizeDirectoryTaskSpec
from ordnung.entities import ToolSecurityPolicy
from ordnung.environment import Environment
from ordnung.llm import LLMClient
from ordnung.tools import CreateDirectoryTool
from ordnung.tools import ListDirectoryTool
from ordnung.tools import MoveFileOrDirectoryTool
from ordnung.tools import ReadBinaryFileTool
from ordnung.tools import ReadTextFileTool
from ordnung.tools import Tool
from ordnung.tui import print_final_result
from ordnung.tui import print_task_spec


def organize(dir_path: Path) -> OrganizeDirectoryResult:
    """
    Organize the files in the specified directory.

    Parameters
    ----------
    dir_path
        The path to the directory to organize.

    Returns
    -------
    Task execution result as reported by the agent.
    """
    dir_path = dir_path.resolve()
    task_spec = OrganizeDirectoryTaskSpec(dir_path=dir_path)
    print_task_spec(task_spec)
    tools: list[type[Tool]] = [
        ListDirectoryTool,
        CreateDirectoryTool,
        MoveFileOrDirectoryTool,
        ReadTextFileTool,
        ReadBinaryFileTool,
    ]
    sec_policy = ToolSecurityPolicy(
        fs_root_jail=dir_path,
        approved_tool_names=[],
    )
    env = Environment(tools=tools, sec_policy=sec_policy)
    llm_client = LLMClient(
        base_url="http://192.168.50.100:11434/v1",
        api_key="ollama",
        model="gpt-oss:20b",
    )
    agent = Agent(llm_client=llm_client, tools=tools)
    result = agent.run_until_done(task_spec=task_spec, env=env)
    print_final_result(result)
    return result
