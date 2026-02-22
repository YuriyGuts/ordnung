from dataclasses import dataclass
from pathlib import Path


@dataclass
class OrganizeDirectoryResult:
    is_success: bool
    error: str | None = None


@dataclass
class OrganizeDirectoryTaskSpec:
    dir_path: Path
