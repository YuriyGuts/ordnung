"""Implements the tools available to the environment/agent."""

import shutil
from abc import ABC
from abc import abstractmethod
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from ordnung.security import ToolSecurityPolicy


class Tool(BaseModel, ABC):
    """The base tool class that represents a specific tool call instance with specific arguments."""

    @classmethod
    def get_name(cls) -> str:
        """Get the name of the tool."""
        return cls.__name__

    @classmethod
    def get_description(cls) -> str:
        """Get the description of the tool."""
        return cls.__doc__ or ""

    @classmethod
    def to_openai_format(cls) -> dict:
        """Format the tool as an OpenAI tool definition."""
        return {
            "type": "function",
            "name": cls.get_name(),
            "description": cls.get_description(),
            "parameters": cls.model_json_schema(),
        }

    @abstractmethod
    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        """Execute the tool."""
        pass


class ListDirectoryTool(Tool):
    """Lists the contents of the specified directory."""

    dir_path: Path = Field(description="The path to the directory to list the contents of.")

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        """Execute the tool."""
        sec_policy.validate_path_access(self.dir_path)
        output_items = []
        for item in self.dir_path.iterdir():
            if item.is_file():
                item_type = "file"
            elif item.is_dir():
                item_type = "directory"
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
        """Execute the tool."""
        sec_policy.validate_path_access(self.dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        return {"created": True}


class MoveFileOrDirectoryTool(Tool):
    """Moves a file or directory to a new location."""

    source_path: Path = Field(description="Source file/directory path.")
    destination_path: Path = Field(description="Destination file/directory path.")

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        """Execute the tool."""
        sec_policy.validate_path_access(self.source_path)
        sec_policy.validate_path_access(self.destination_path)
        shutil.move(src=self.source_path, dst=self.destination_path)
        return {"moved": True}


class ReadTextFileTool(Tool):
    """Reads the contents of the specified file, assuming it is UTF-8-encoded text."""

    file_path: Path = Field(description="The path of the file to read.")

    def run(self, sec_policy: ToolSecurityPolicy) -> dict:
        """Execute the tool."""
        sec_policy.validate_path_access(self.file_path)
        file_contents = self.file_path.read_text(encoding="utf-8")
        return {"contents": file_contents}


class ReadBinaryFileTool(Tool):
    """Reads the contents of a binary file as a hex string."""

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
        """Execute the tool."""
        if self.limit > self._MAX_LIMIT:
            raise RuntimeError(
                f"The specified limit is greater than the maximum allowed limit ({self._MAX_LIMIT})"
            )
        sec_policy.validate_path_access(self.file_path)

        with open(self.file_path, mode="rb") as fd:
            fd.seek(self.offset)
            file_contents = fd.read(self.limit)

        return {"hex_contents": file_contents.hex(sep=":")}
