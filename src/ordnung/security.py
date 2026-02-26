"""Implements the security features of the application."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class ToolSecurityPolicy:
    """Represents the security configuration for tool execution."""

    # The root path the agent is allowed to operate within.
    # Any attempts to perform file operations outside this path
    # must be rejected by the tool implementations.
    fs_root_jail: Path

    # The names of the tools the user has auto-approved in the current session.
    approved_tool_names: set[str] = field(default_factory=set)

    def validate_path_access(self, path: Path) -> None:
        """Check whether the tool can access the specified path according to the security policy."""
        path = path.resolve()
        if not path.is_relative_to(self.fs_root_jail):
            msg = (
                f"The requested path ({path}) is outside "
                f"the task root path ({self.fs_root_jail}). "
                "Rejected by security policy."
            )
            raise RuntimeError(msg)
