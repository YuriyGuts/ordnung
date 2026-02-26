import shutil
from abc import abstractmethod
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from ordnung.entities import ToolSecurityPolicy


class Tool(BaseModel):
    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    @classmethod
    def get_description(cls) -> str:
        return cls.__doc__ or ""

    @classmethod
    def _check_fs_jail(cls, path: Path, sec_policy: ToolSecurityPolicy) -> None:
        path = path.resolve()
        if not path.is_relative_to(sec_policy.fs_root_jail):
            msg = (
                f"The requested path ({path}) is outside "
                f"the task root path ({sec_policy.fs_root_jail}). "
                "Rejected by security policy."
            )
            raise RuntimeError(msg)

    @abstractmethod
    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        pass


class ListDirectoryTool(Tool):
    """Lists the contents of the specified directory."""

    dir_path: Path = Field(description="The path to the directory to list the contents of.")

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        self._check_fs_jail(self.dir_path, sec_policy)
        output_items = []
        for item in self.dir_path.iterdir():
            if item.is_file():
                item_type = "file"
            elif item.is_dir():
                item_type = "directory"
            elif item.is_symlink():
                item_type = "symlink"
            else:
                item_type = "unknown"

            item_output = {
                "type": item_type,
                "name": item.name,
                "size_bytes": item.stat().st_size,
            }
            output_items.append(item_output)

        return {"output": output_items}


class CreateDirectoryTool(Tool):
    """Creates a directory at the specified path, creating any parent directories if needed."""

    dir_path: Path = Field(description="The path to the desired directory.")

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        self._check_fs_jail(self.dir_path, sec_policy)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        return {"created": True}


class MoveFileOrDirectoryTool(Tool):
    """Move a file or directory to a new location."""

    source_path: Path = Field(description="Source file/directory path.")
    destination_path: Path = Field(description="Destination file/directory path.")

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        self._check_fs_jail(self.source_path, sec_policy)
        self._check_fs_jail(self.destination_path, sec_policy)
        shutil.move(src=self.source_path, dst=self.destination_path)
        return {"moved": True}


class ReadTextFileTool(Tool):
    """Read the contents of the specified file, assuming it is UTF8-encoded text."""

    file_path: Path = Field(description="The path of the file to read.")

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        self._check_fs_jail(self.file_path, sec_policy)
        file_contents = self.file_path.read_text(encoding="utf-8")
        return {"contents": file_contents}


class ReadBinaryFileTool(Tool):
    """Read the contents of a binary file as a hex string."""

    _MAX_LIMIT = 1024

    file_path: Path = Field(description="The path of the file to read.")
    offset: int = Field(
        default=0,
        description="The number of the start byte to read from (default 0).",
    )
    limit: int = Field(
        default=256,
        description=f"The number of bytes to read starting from the offset (max {_MAX_LIMIT}).",
    )

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        if self.limit > self._MAX_LIMIT:
            raise RuntimeError(
                f"The specified limit is greater than the maximum allowed limit ({self._MAX_LIMIT})"
            )
        self._check_fs_jail(self.file_path, sec_policy)
        file_contents = self.file_path.read_bytes()
        hex_contents = file_contents[self.offset : self.limit].hex(sep=":")
        return {"hex_contents": hex_contents}
