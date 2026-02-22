from pathlib import Path

from ordnung.agent import Agent
from ordnung.entities import OrganizeDirectoryTaskSpec
from ordnung.environment import Environment
from ordnung.llm import LLMClient
from ordnung.tools import ListDirectoryTool
from ordnung.tools import Tool


def organize(dir_path: Path) -> None:
    print(f"Organizing directory: {dir_path}")
    tools: list[type[Tool]] = [ListDirectoryTool]
    env = Environment(tools=tools)
    llm_client = LLMClient(
        base_url="http://192.168.50.100:11434/v1", api_key="ollama", model="gpt-oss:20b"
    )
    agent = Agent(llm_client=llm_client, tools=tools)
    task_spec = OrganizeDirectoryTaskSpec(dir_path=dir_path)
    result = agent.run_until_done(task_spec=task_spec, env=env)
    print(result)
