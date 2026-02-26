from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class OrganizeDirectoryResult:
    is_success: bool
    error: str | None = None


@dataclass
class OrganizeDirectoryTaskSpec:
    dir_path: Path


@dataclass
class ToolSecurityPolicy:
    fs_root_jail: Path
    approved_tool_names: list[str] = field(default_factory=list)
