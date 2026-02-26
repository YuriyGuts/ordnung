"""Tests for ordnung.environment."""

import json

import pytest

from ordnung.environment import Environment
from ordnung.security import ToolSecurityPolicy
from ordnung.tools import CreateDirectoryTool
from ordnung.tools import ListDirectoryTool
from ordnung.tools import ReadTextFileTool


@pytest.fixture
def env(sec_policy):
    return Environment(
        tools=[ListDirectoryTool, CreateDirectoryTool, ReadTextFileTool],
        sec_policy=sec_policy,
    )


@pytest.fixture
def auto_approved_env(sandbox):
    """An environment where all tools are pre-approved (no TUI prompts)."""
    policy = ToolSecurityPolicy(
        fs_root_jail=sandbox,
        approved_tool_names={"ListDirectoryTool", "CreateDirectoryTool", "ReadTextFileTool"},
    )
    return Environment(
        tools=[ListDirectoryTool, CreateDirectoryTool, ReadTextFileTool],
        sec_policy=policy,
    )


def test_run_tool_unknown_tool(auto_approved_env):
    # GIVEN a tool name that doesn't exist
    # WHEN running it
    result = auto_approved_env.run_tool("NonExistentTool", "{}")

    # THEN an error is returned
    assert "error" in result
    assert "does not exist" in result["error"]


def test_run_tool_invalid_json_args(auto_approved_env):
    # GIVEN invalid JSON as arguments
    # WHEN running a tool
    result = auto_approved_env.run_tool("ListDirectoryTool", "not json")

    # THEN an error is returned
    assert "error" in result
    assert "Invalid arguments" in result["error"]


def test_run_tool_missing_required_args(auto_approved_env):
    # GIVEN missing required arguments
    # WHEN running a tool
    result = auto_approved_env.run_tool("ListDirectoryTool", "{}")

    # THEN an error is returned
    assert "error" in result
    assert "Invalid arguments" in result["error"]


def test_run_tool_successful_execution(auto_approved_env, sandbox):
    # GIVEN a valid tool call with a valid directory
    args = json.dumps({"dir_path": str(sandbox)})

    # WHEN running the tool
    result = auto_approved_env.run_tool("ListDirectoryTool", args)

    # THEN it returns tool output (not an error)
    assert "error" not in result
    assert "output" in result


def test_run_tool_catches_runtime_errors(auto_approved_env, sandbox):
    # GIVEN a tool call that will fail at runtime (reading a nonexistent file)
    args = json.dumps({"file_path": str(sandbox / "nonexistent.txt")})

    # WHEN running the tool
    result = auto_approved_env.run_tool("ReadTextFileTool", args)

    # THEN the runtime error is caught and returned
    assert "error" in result
    assert "Tool call failed" in result["error"]


def test_run_tool_with_approval_yes(env, sandbox, monkeypatch):
    # GIVEN a non-approved tool and user approves with "y"
    monkeypatch.setattr("ordnung.environment.approval_prompt", lambda: "y")
    args = json.dumps({"dir_path": str(sandbox)})

    # WHEN running the tool
    result = env.run_tool("ListDirectoryTool", args)

    # THEN the tool executes successfully
    assert "error" not in result
    assert "output" in result


def test_run_tool_with_approval_no(env, sandbox, monkeypatch):
    # GIVEN a non-approved tool and user rejects with "n"
    monkeypatch.setattr("ordnung.environment.approval_prompt", lambda: "n")
    args = json.dumps({"dir_path": str(sandbox)})

    # WHEN running the tool
    result = env.run_tool("ListDirectoryTool", args)

    # THEN the tool call is rejected
    assert result == {"error": "Tool call rejected by user"}


def test_run_tool_with_approval_quit(env, sandbox, monkeypatch):
    # GIVEN a non-approved tool and user quits with "q"
    monkeypatch.setattr("ordnung.environment.approval_prompt", lambda: "q")
    args = json.dumps({"dir_path": str(sandbox)})

    # WHEN running the tool
    # THEN a KeyboardInterrupt is raised
    with pytest.raises(KeyboardInterrupt):
        env.run_tool("ListDirectoryTool", args)


def test_run_tool_with_approval_always(env, sandbox, monkeypatch):
    # GIVEN a non-approved tool and user approves-all with "a"
    monkeypatch.setattr("ordnung.environment.approval_prompt", lambda: "a")
    args = json.dumps({"dir_path": str(sandbox)})

    # WHEN running the tool
    result = env.run_tool("ListDirectoryTool", args)

    # THEN the tool executes and is added to auto-approved list
    assert "error" not in result
    assert "ListDirectoryTool" in env.sec_policy.approved_tool_names


def test_run_tool_with_approval_feedback(env, sandbox, monkeypatch):
    # GIVEN a non-approved tool and user provides feedback
    monkeypatch.setattr("ordnung.environment.approval_prompt", lambda: "f")
    monkeypatch.setattr("ordnung.environment.user_feedback_prompt", lambda: "Try a different dir")
    args = json.dumps({"dir_path": str(sandbox)})

    # WHEN running the tool
    result = env.run_tool("ListDirectoryTool", args)

    # THEN the feedback is returned as an error
    assert "error" in result
    assert "Try a different dir" in result["error"]


def test_tool_registry_populated(env):
    # GIVEN an environment with registered tools
    # THEN the registry contains all tool names
    assert set(env.tool_registry.keys()) == {
        "ListDirectoryTool",
        "CreateDirectoryTool",
        "ReadTextFileTool",
    }
