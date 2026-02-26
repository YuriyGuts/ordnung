"""Common entity classes for the application."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class OrganizeDirectoryTaskSpec:
    """Represents the input parameters of the task."""

    # The path to the directory to organize.
    dir_path: Path


@dataclass
class OrganizeDirectoryResult:
    """Represents the results of performing the task."""

    # Whether the agent execution was successful.
    is_success: bool

    # If specified, the error message returned by the agent.
    error: str | None = None


@dataclass
class ToolSecurityPolicy:
    """Represents the security configuration for tool execution."""

    # The root path the agent is allowed to operate within.
    # Any attempts to perform file operations outside this path
    # must be rejected by the tool implementations.
    fs_root_jail: Path

    # The names of the tools the user has auto-approved in the current session.
    approved_tool_names: list[str] = field(default_factory=list)
