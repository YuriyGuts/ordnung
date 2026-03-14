"""Implements the top-level operations for directory organization."""

from pathlib import Path

from ordnung.agent import Agent
from ordnung.entities import DEFAULT_MAX_ITERATIONS
from ordnung.entities import LLMAPIMode
from ordnung.entities import OrganizeDirectoryResult
from ordnung.entities import OrganizeDirectoryTaskSpec
from ordnung.environment import Environment
from ordnung.llm import create_llm_client
from ordnung.security import ToolSecurityPolicy
from ordnung.tools import CreateDirectoryTool
from ordnung.tools import ListDirectoryTool
from ordnung.tools import MoveFileOrDirectoryTool
from ordnung.tools import ReadBinaryFileTool
from ordnung.tools import ReadTextFileTool
from ordnung.tools import Tool
from ordnung.tui import print_final_result
from ordnung.tui import print_llm_api_details
from ordnung.tui import print_skip_permissions_warning
from ordnung.tui import print_task_spec


def organize(
    dir_path: Path,
    llm_api_base_url: str,
    llm_api_key: str,
    llm_name: str,
    llm_api_mode: LLMAPIMode = LLMAPIMode.RESPONSES,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    skip_permissions: bool = False,
) -> OrganizeDirectoryResult:
    """
    Organize the files in the specified directory.

    Parameters
    ----------
    dir_path
        The path to the directory to organize.
    llm_api_base_url
        The base URL of the OpenAI-compatible LLM API.
    llm_api_key
        The API key for LLM API authentication.
    llm_name
        The name of the LLM to use in the LLM API.
    llm_api_mode
        The LLM API mode to use.
    max_iterations
        The maximum allowed number of agentic loop iterations before aborting.
    skip_permissions
        Pre-approve all tools so the agent runs without interactive permission prompts.

    Returns
    -------
    Task execution result as reported by the agent.
    """
    dir_path = dir_path.resolve()

    # Specify the task for the agent.
    task_spec = OrganizeDirectoryTaskSpec(dir_path=dir_path)
    print_task_spec(task_spec)

    # Discover the tools.
    tools: list[type[Tool]] = [
        ListDirectoryTool,
        CreateDirectoryTool,
        MoveFileOrDirectoryTool,
        ReadTextFileTool,
        ReadBinaryFileTool,
    ]

    # Set up the environment.
    sec_policy = ToolSecurityPolicy(fs_root_jail=dir_path)
    if skip_permissions:
        print_skip_permissions_warning()
        sec_policy.approved_tool_names |= {tool.get_name() for tool in tools}
    env = Environment(tools=tools, sec_policy=sec_policy)

    # Create the agent.
    llm_client = create_llm_client(
        api_mode=llm_api_mode,
        base_url=llm_api_base_url,
        api_key=llm_api_key,
        model=llm_name,
    )
    print_llm_api_details(llm_api_base_url, llm_name)
    agent = Agent(llm_client=llm_client, tools=tools)

    # Run the agent.
    result = agent.run_until_done(task_spec=task_spec, env=env, max_iterations=max_iterations)
    print_final_result(result)

    return result
