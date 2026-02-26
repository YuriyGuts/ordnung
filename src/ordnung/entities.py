"""Common entity classes for the application."""

from dataclasses import dataclass
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
